fn main() {
    let version = match std::env::var("AGENT_MEMORY_VERSION") {
        Ok(v) => {
            let v = v.trim().to_string();
            match v.strip_prefix('v') {
                Some(stripped) => stripped.to_string(),
                None => v,
            }
        }
        Err(_) => std::env::var("CARGO_PKG_VERSION").unwrap(),
    };
    println!("cargo:rustc-env=AGENT_MEMORY_VERSION={version}");
    println!("cargo:rerun-if-env-changed=AGENT_MEMORY_VERSION");
}
