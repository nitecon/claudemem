//! Device resolution for the candle backend.
//!
//! The local inference path needs a concrete [`candle_core::Device`] before it
//! can load weights or run a forward pass. Three knobs drive resolution:
//!
//! 1. The persisted preference `local.device` in `dream.toml`
//!    (`auto | cpu | metal | cuda`).
//! 2. Platform detection at runtime (Apple Silicon → Metal first; NVIDIA with
//!    CUDA → CUDA first; everything else → CPU).
//! 3. Initialization fallbacks — if Metal or CUDA can't initialize (no
//!    GPU/driver, headless CI box), we log and drop to CPU so the dream pass
//!    still runs.
//!
//! The module is deliberately small and dependency-light so both the real
//! candle backend and unit tests can call it without dragging in weights or
//! network I/O.

use candle_core::Device;

use crate::inference::InferenceError;

/// Human-readable device preference, parsed from `local.device` in `dream.toml`
/// and from the eventual `--device` CLI flag (not wired yet — the knob lives
/// in settings today and CLI exposure is a straightforward follow-up).
///
/// `Auto` applies the platform heuristic:
///   * macOS aarch64 → Metal (fallback CPU on init failure)
///   * CUDA-enabled build → CUDA (fallback CPU on init failure)
///   * Everything else → CPU
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DevicePreference {
    #[default]
    Auto,
    Cpu,
    Metal,
    Cuda,
}

impl DevicePreference {
    /// Stable short string used in `dream.toml` and CLI flags.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }
}

impl std::str::FromStr for DevicePreference {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            other => Err(format!(
                "invalid device preference '{other}' (expected auto|cpu|metal|cuda)"
            )),
        }
    }
}

/// Resolve a [`DevicePreference`] into a concrete [`candle_core::Device`].
///
/// The `Auto` branch applies the platform heuristic and falls back to CPU
/// silently when the preferred accelerator is not available — users on a
/// cold laptop can run the dream pipeline without tripping errors.
///
/// Explicit preferences (`Cpu`, `Metal`, `Cuda`) are strict — if the user
/// asked for Metal and Metal isn't available, we return
/// [`InferenceError::DeviceUnavailable`] rather than silently downgrading.
/// A silent downgrade on an explicit choice would defeat the purpose of the
/// knob.
pub fn resolve_device(pref: DevicePreference) -> Result<Device, InferenceError> {
    match pref {
        DevicePreference::Cpu => Ok(Device::Cpu),
        DevicePreference::Metal => new_metal_strict(),
        DevicePreference::Cuda => new_cuda_strict(),
        DevicePreference::Auto => Ok(resolve_auto()),
    }
}

/// Auto-resolution. Tries the best-available accelerator for the build and
/// silently falls back to CPU on any initialization error. Emits a tracing
/// debug line so operators can tell which device was picked without having
/// to read through the source.
fn resolve_auto() -> Device {
    // macOS aarch64 → prefer Metal. Intel macs and other x86_64 platforms
    // fall through to the CUDA probe (no-op in non-cuda builds) and finally
    // to CPU.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        match Device::new_metal(0) {
            Ok(d) => {
                tracing::debug!("auto-device: using Metal(0) on Apple Silicon");
                return d;
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "auto-device: Metal init failed on Apple Silicon; falling back to CPU"
                );
            }
        }
    }

    // Any platform: try CUDA. `Device::new_cuda` returns an error at compile
    // time if the `cuda` feature isn't enabled on candle-core, but at
    // runtime on a non-CUDA host it surfaces a clean error we can skip past.
    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(0) {
            Ok(d) => {
                tracing::debug!("auto-device: using CUDA(0)");
                return d;
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "auto-device: CUDA init failed; falling back to CPU"
                );
            }
        }
    }

    tracing::debug!("auto-device: using CPU");
    Device::Cpu
}

/// Strict Metal resolution. Surfaces [`InferenceError::DeviceUnavailable`]
/// when Metal can't initialize so the user's explicit preference is honored
/// (no silent downgrade).
fn new_metal_strict() -> Result<Device, InferenceError> {
    // Keep the cfg-guarded path so non-macOS builds don't ship Metal code.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        Device::new_metal(0).map_err(|e| InferenceError::DeviceUnavailable {
            requested: "metal".to_string(),
            detail: e.to_string(),
        })
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        Err(InferenceError::DeviceUnavailable {
            requested: "metal".to_string(),
            detail: "Metal is only available on macOS aarch64".to_string(),
        })
    }
}

/// Strict CUDA resolution. Symmetric with [`new_metal_strict`].
fn new_cuda_strict() -> Result<Device, InferenceError> {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0).map_err(|e| InferenceError::DeviceUnavailable {
            requested: "cuda".to_string(),
            detail: e.to_string(),
        })
    }

    #[cfg(not(feature = "cuda"))]
    {
        Err(InferenceError::DeviceUnavailable {
            requested: "cuda".to_string(),
            detail: "binary was built without the candle `cuda` feature".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preference_round_trips_through_string() {
        for p in [
            DevicePreference::Auto,
            DevicePreference::Cpu,
            DevicePreference::Metal,
            DevicePreference::Cuda,
        ] {
            let parsed: DevicePreference = p.as_str().parse().unwrap();
            assert_eq!(parsed, p);
        }
    }

    #[test]
    fn preference_rejects_unknown() {
        assert!("tpu".parse::<DevicePreference>().is_err());
    }

    #[test]
    fn forced_cpu_always_resolves_to_cpu() {
        // Regardless of host architecture, `Cpu` is always available.
        let d = resolve_device(DevicePreference::Cpu).expect("cpu must resolve");
        assert!(matches!(d, Device::Cpu));
    }

    /// On a CPU-only test host (CI), the strict Metal branch either returns
    /// a real Metal device (mac ARM workstation) or a clean
    /// `DeviceUnavailable` — never a panic.
    #[test]
    fn forced_metal_never_panics() {
        let _ = resolve_device(DevicePreference::Metal);
    }

    /// Auto must always return *some* device — the CPU fallback is the floor.
    #[test]
    fn auto_always_resolves_to_some_device() {
        let d = resolve_device(DevicePreference::Auto).expect("auto must resolve");
        // Any variant is acceptable — we just assert construction succeeded.
        let _ = d;
    }

    /// On macOS aarch64 specifically, auto resolution MUST succeed and must
    /// land on Metal (or CPU if Metal init genuinely failed). This test is
    /// only meaningful on Apple Silicon — elsewhere it degenerates to the
    /// same assertion as the previous test.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn auto_picks_metal_on_apple_silicon_when_available() {
        let d = resolve_device(DevicePreference::Auto).expect("auto must resolve");
        // We don't require Metal specifically — a CI box with Metal disabled
        // will still resolve to CPU. The constraint is "resolved to something".
        match d {
            Device::Cpu | Device::Metal(_) => {}
            other => panic!("unexpected device on apple silicon: {other:?}"),
        }
    }
}
