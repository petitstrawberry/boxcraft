//! Native controller snapshots, independent of pointer capture and UI navigation.

use std::collections::HashMap;

use boxcraft_core::PlayerInput;
use scarlet_ui::{GamepadButton, GamepadEvent};

#[derive(Default)]
pub struct Controls {
    devices: HashMap<u32, GamepadEvent>,
}

impl Controls {
    /// Update one device and return newly pressed buttons. Disconnect/focus
    /// resets release that device without losing another controller's state.
    pub fn update(&mut self, event: GamepadEvent) -> u32 {
        let previous = self.buttons();
        if event.reset {
            self.devices.remove(&event.device_id);
            return 0;
        }
        self.devices.insert(event.device_id, event);
        self.buttons() & !previous
    }

    pub fn clear(&mut self) {
        self.devices.clear();
    }

    fn buttons(&self) -> u32 {
        self.devices.values().fold(0, |buttons, state| {
            buttons
                | state.buttons
                | if state.left_trigger > 16383 {
                    bit(GamepadButton::LeftTrigger)
                } else {
                    0
                }
                | if state.right_trigger > 16383 {
                    bit(GamepadButton::RightTrigger)
                } else {
                    0
                }
        })
    }

    /// Return analog movement and look speed. Dead zones are radial; remaining
    /// travel is rescaled so a small tilt moves slowly and diagonals stay bounded.
    pub fn sample(&self) -> (PlayerInput, [f32; 2]) {
        let mut movement = [0.0f32; 2];
        let mut look = [0.0f32; 2];
        for state in self.devices.values() {
            let left = stick(state.left_x, state.left_y);
            let right = stick(state.right_x, state.right_y);
            for axis in 0..2 {
                if left[axis].abs() > movement[axis].abs() {
                    movement[axis] = left[axis];
                }
                if right[axis].abs() > look[axis].abs() {
                    look[axis] = right[axis];
                }
            }
        }
        (
            PlayerInput {
                strafe_axis: movement[0],
                forward_axis: -movement[1],
                jump: self.buttons() & bit(GamepadButton::South) != 0,
                ..PlayerInput::default()
            },
            look,
        )
    }
}

pub const fn bit(button: GamepadButton) -> u32 {
    1 << button as u8
}

fn stick(x: i16, y: i16) -> [f32; 2] {
    const DEAD_ZONE: f32 = 0.18;
    let x = f32::from(x) / 32767.0;
    let y = f32::from(y) / 32767.0;
    let length = (x * x + y * y).sqrt();
    if length <= DEAD_ZONE {
        return [0.0; 2];
    }
    let scale = ((length.min(1.0) - DEAD_ZONE) / (1.0 - DEAD_ZONE)) / length;
    [x * scale, y * scale]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_drift_half_tilt_and_diagonals() {
        assert_eq!(stick(1500, -2000), [0.0; 2]);
        let half = stick(16384, 0);
        assert!(half[0] > 0.3 && half[0] < 0.5);
        let diagonal = stick(i16::MIN, i16::MAX);
        assert!((diagonal[0].powi(2) + diagonal[1].powi(2) - 1.0).abs() < 0.0001);
    }

    #[test]
    fn actions_are_edges_and_reset_stops_motion() {
        let mut controls = Controls::default();
        let state = GamepadEvent {
            left_y: -32767,
            right_x: 32767,
            left_trigger: 32767,
            buttons: bit(GamepadButton::South),
            ..GamepadEvent::default()
        };
        assert_eq!(
            controls.update(state),
            bit(GamepadButton::South) | bit(GamepadButton::LeftTrigger)
        );
        assert_eq!(controls.update(state), 0);
        let (movement, look) = controls.sample();
        assert_eq!(movement.forward_axis, 1.0);
        assert!(movement.jump);
        assert_eq!(look, [1.0, 0.0]);
        controls.update(GamepadEvent {
            reset: true,
            ..state
        });
        assert_eq!(controls.sample(), (PlayerInput::default(), [0.0; 2]));
    }

    #[test]
    fn one_device_reset_preserves_the_other_and_focus_clear_releases_all() {
        let mut controls = Controls::default();
        controls.update(GamepadEvent {
            device_id: 1,
            left_x: 32767,
            ..GamepadEvent::default()
        });
        controls.update(GamepadEvent {
            device_id: 2,
            right_y: -32767,
            ..GamepadEvent::default()
        });
        controls.update(GamepadEvent {
            device_id: 1,
            reset: true,
            ..GamepadEvent::default()
        });
        assert_eq!(controls.sample(), (PlayerInput::default(), [0.0, -1.0]));
        controls.clear();
        assert_eq!(controls.sample(), (PlayerInput::default(), [0.0; 2]));
    }
}
