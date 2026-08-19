//! Boxcraft application entry point.

mod mesh_worker;

mod ui;

fn main() {
    if let Err(error) = ui::run() {
        eprintln!("boxcraft: {error}");
    }
}
