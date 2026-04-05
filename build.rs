#[cfg(windows)]
use vcpkg;

#[cfg(windows)]
fn find_package(name: &str) {
    vcpkg::Config::new()
        .target_triplet("x64-windows")
        .find_package(&name)
        .unwrap();
}

#[cfg(not(windows))]
fn find_package(_name: &str) {}

fn main() {
    let target = std::env::var("TARGET").unwrap();

    if target.contains("windows") {
        find_package("openblas");
        find_package("lapack-reference");
    }
}
