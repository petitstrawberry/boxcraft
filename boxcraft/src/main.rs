//! Boxcraft application entry point.

#[cfg(target_os = "scarlet")]
mod ui;

#[cfg(target_os = "scarlet")]
fn main() {
    if let Err(error) = ui::run() {
        eprintln!("boxcraft: {error}");
    }
}

#[cfg(not(target_os = "scarlet"))]
fn main() {
    eprintln!("boxcraft: the graphical application is available on Scarlet OS only");
}
