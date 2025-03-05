use bevy::math::Vec2;

const CORNER: f32 = 0.2;

// A box with a cross through it
pub const UNKNOWN: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 1.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 1.0),
    Vec2::new(0.0, 1.0),
];

pub const SPACE: &[Vec2] = &[];

const UPPER_A: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(0.5, 0.0),
    Vec2::new(1.0, 1.0),
    Vec2::new(0.67, 0.5),
    Vec2::new(0.33, 0.5),
    Vec2::new(0.0, 1.0),
];

const UPPER_B: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0 - CORNER, 0.0),
    Vec2::new(1.0, CORNER),
    Vec2::new(1.0, 0.5 - CORNER),
    Vec2::new(1.0 - CORNER, 0.5),
    Vec2::new(1.0, 0.5 + CORNER),
    Vec2::new(1.0, 1.0 - CORNER),
    Vec2::new(1.0 - CORNER, 1.0),
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.5),
    Vec2::new(1.0 - CORNER, 0.5),
    Vec2::new(0.0, 0.5),
    Vec2::new(0.0, 1.0),
];

const UPPER_C: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(0.0, 1.0),
    Vec2::new(1.0, 1.0),
];

const UPPER_D: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0 - CORNER, 0.0),
    Vec2::new(1.0, CORNER),
    Vec2::new(1.0, 1. - CORNER),
    Vec2::new(1.0 - CORNER, 1.0),
    Vec2::new(0.0, 1.0),
];

const UPPER_E: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(0.0, 0.5),
    Vec2::new(1.0 - CORNER, 0.5),
    Vec2::new(0.0, 0.5),
    Vec2::new(0.0, 1.0),
    Vec2::new(1.0, 1.0),
];

const UPPER_F: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(0.0, 0.5),
    Vec2::new(1.0 - CORNER, 0.5),
    Vec2::new(0.0, 0.5),
    Vec2::new(0.0, 1.0),
];

const UPPER_G: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(0.0, 1.0),
    Vec2::new(0.5, 1.0),
    Vec2::new(1.0 - CORNER, 0.5),
    Vec2::new(1.0 - CORNER, 1.0),
];

const UPPER_H: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(0.0, 0.5),
    Vec2::new(1.0, 0.5),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 1.0),
];

const UPPER_I: &[Vec2] = &[
    Vec2::new(0.5, 1.0),
    Vec2::new(0.25, 1.0),
    Vec2::new(0.5, 1.0),
    Vec2::new(0.5, 0.0),
    Vec2::new(0.25, 0.0),
    Vec2::new(0.75, 0.0),
    Vec2::new(0.5, 0.0),
    Vec2::new(0.5, 1.0),
    Vec2::new(0.75, 1.0),
];

const UPPER_J: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(0.5, 1.0),
    Vec2::new(0.5, 0.0),
    Vec2::new(CORNER, 0.0),
    Vec2::new(1.0 - CORNER, 0.0),
    Vec2::new(0.5, 0.0),
    Vec2::new(0.5, 1.0),
];

const UPPER_K: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(0.0, 0.5),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.0, 0.5),
    Vec2::new(1.0, 1.0),
];

const UPPER_L: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(0.0, 1.0),
    Vec2::new(1.0, 1.0),
];

const UPPER_M: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(0.5, 1.0 - CORNER),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 1.0),
];

const UPPER_N: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 1.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 1.0),
];

const UPPER_O: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 1.0),
    Vec2::new(0.0, 1.0),
];

const UPPER_P: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 0.5),
    Vec2::new(0.0, 0.5),
    Vec2::new(0.0, 1.0),
];

const UPPER_Q: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 0.5),
    Vec2::new(0.75, 0.75),
    Vec2::new(0.5, 0.5),
    Vec2::new(1.0, 1.0),
    Vec2::new(0.75, 0.75),
    Vec2::new(0.5, 1.0),
    Vec2::new(0.0, 1.0),
];

const UPPER_R: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0 - CORNER, 0.0),
    Vec2::new(1.0, CORNER),
    Vec2::new(1.0, 0.5 - CORNER),
    Vec2::new(1.0 - CORNER, 0.5),
    Vec2::new(0.0, 0.5),
    Vec2::new(0.5, 0.5),
    Vec2::new(1.0, 1.0),
    Vec2::new(0.5, 0.5),
    Vec2::new(0.0, 0.5),
    Vec2::new(0.0, 1.0),
];

const UPPER_S: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(1.0, 1.0),
    Vec2::new(1.0, 0.5),
    Vec2::new(0.0, 0.5),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(0.0, 0.5),
    Vec2::new(1.0, 0.5),
    Vec2::new(1.0, 1.0),
    Vec2::new(0.0, 1.0),
];

const UPPER_T: &[Vec2] = &[
    Vec2::new(0.5, 1.0),
    Vec2::new(0.5, 0.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.5, 0.0),
    Vec2::new(0.5, 1.0),
];

const UPPER_U: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(0.0, 1.0),
    Vec2::new(1.0, 1.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 1.0),
];

const UPPER_V: &[Vec2] = &[
    Vec2::new(0.5, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(0.5, 1.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.5, 1.0),
];

const UPPER_W: &[Vec2] = &[
    Vec2::new(0.33, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(0.33, 1.0),
    Vec2::new(0.5, 0.5),
    Vec2::new(0.67, 1.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.67, 1.0),
];

const UPPER_X: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(0.5, 0.5),
    Vec2::new(0.0, 0.0),
    Vec2::new(0.5, 0.5),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.5, 0.5),
    Vec2::new(1.0, 1.0),
    Vec2::new(0.5, 0.5),
    Vec2::new(0.0, 1.0),
];

const UPPER_Y: &[Vec2] = &[
    Vec2::new(0.5, 1.0),
    Vec2::new(0.5, 0.5),
    Vec2::new(0.0, 0.0),
    Vec2::new(0.5, 0.5),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.5, 0.5),
    Vec2::new(0.5, 1.0),
];

const UPPER_Z: &[Vec2] = &[
    Vec2::new(1.0, 1.0),
    Vec2::new(0.0, 1.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.0, 1.0),
    Vec2::new(1.0, 1.0),
];

const DIGIT_0: &[Vec2] = &[
    Vec2::new(1.0, 1.0),
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 1.0),
];

const DIGIT_1: &[Vec2] = &[
    Vec2::new(0.5, 1.0),
    Vec2::new(0.5, 0.0),
    Vec2::new(0.5 - CORNER, CORNER),
    Vec2::new(0.5, 0.0),
    Vec2::new(0.5, 1.0),
];

const DIGIT_2: &[Vec2] = &[
    Vec2::new(1.0, 1.0),
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.5),
    Vec2::new(1.0, 0.5),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 0.5),
    Vec2::new(0.0, 0.5),
    Vec2::new(0.0, 1.0),
    Vec2::new(1.0, 1.0),
];

const DIGIT_3: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(1.0 - CORNER, 1.0),
    Vec2::new(1.0, 1.0 - CORNER),
    Vec2::new(1.0, 0.5 + CORNER),
    Vec2::new(1.0 - CORNER, 0.5),
    Vec2::new(0.5, 0.5),
    Vec2::new(1.0 - CORNER, 0.5),
    Vec2::new(1.0, 0.5 - CORNER),
    Vec2::new(1.0, CORNER),
    Vec2::new(1.0 - CORNER, 0.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0 - CORNER, 0.0),
    Vec2::new(1.0, CORNER),
    Vec2::new(1.0, 0.5 - CORNER),
    Vec2::new(1.0 - CORNER, 0.5),
    Vec2::new(0.5, 0.5),
    Vec2::new(1.0 - CORNER, 0.5),
    Vec2::new(1.0, 0.5 + CORNER),
    Vec2::new(1.0, 1.0 - CORNER),
    Vec2::new(1.0 - CORNER, 1.0),
    Vec2::new(0.0, 1.0),
];

const DIGIT_4: &[Vec2] = &[
    Vec2::new(0.75, 1.0),
    Vec2::new(0.75, 0.0),
    Vec2::new(0.0, 0.75),
    Vec2::new(1.0, 0.75),
    Vec2::new(0.75, 0.75),
    Vec2::new(0.75, 1.0),
];

const DIGIT_5: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(1.0 - CORNER, 1.0),
    Vec2::new(1.0, 1.0 - CORNER),
    Vec2::new(1.0, 0.5 + CORNER),
    Vec2::new(1.0 - CORNER, 0.5),
    Vec2::new(0.0, 0.5),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0 - CORNER, 0.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(0.0, 0.5),
    Vec2::new(1.0 - CORNER, 0.5),
    Vec2::new(1.0, 0.5 + CORNER),
    Vec2::new(1.0, 1.0 - CORNER),
    Vec2::new(1.0 - CORNER, 1.0),
    Vec2::new(0.0, 1.0),
];

const DIGIT_6: &[Vec2] = &[
    Vec2::new(1.0, 1.0),
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(0.0, 0.5),
    Vec2::new(1.0, 0.5),
    Vec2::new(1.0, 1.0),
];

const DIGIT_7: &[Vec2] = &[
    Vec2::new(0.5, 1.0),
    Vec2::new(0.5, 0.5),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.5, 0.5),
    Vec2::new(0.5, 1.0),
];

const DIGIT_8: &[Vec2] = &[
    Vec2::new(CORNER, 1.0),
    Vec2::new(0.0, 1.0 - CORNER),
    Vec2::new(0.0, 0.5 + CORNER),
    Vec2::new(CORNER, 0.5),
    Vec2::new(1.0 - CORNER, 0.5),
    Vec2::new(1.0, 0.5 - CORNER),
    Vec2::new(1.0, CORNER),
    Vec2::new(1.0 - CORNER, 0.0),
    Vec2::new(CORNER, 0.0),
    Vec2::new(0.0, CORNER),
    Vec2::new(0.0, 0.5 - CORNER),
    Vec2::new(CORNER, 0.5),
    Vec2::new(1.0 - CORNER, 0.5),
    Vec2::new(1.0, 0.5 + CORNER),
    Vec2::new(1.0, 1.0 - CORNER),
    Vec2::new(1.0 - CORNER, 1.0),
    Vec2::new(CORNER, 1.0),
];

const DIGIT_9: &[Vec2] = &[
    Vec2::new(1.0, 1.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(0.0, 0.5),
    Vec2::new(1.0, 0.5),
    Vec2::new(1.0, 1.0),
];

const PERIOD: &[Vec2] = &[
    Vec2::new(0.4, 1.0),
    Vec2::new(0.4, 0.8),
    Vec2::new(0.6, 0.8),
    Vec2::new(0.6, 1.0),
    Vec2::new(0.4, 1.0),
];

const COLON: &[Vec2] = &[
    Vec2::new(0.4, 1.0),
    Vec2::new(0.4, 0.8),
    Vec2::new(0.6, 0.8),
    Vec2::new(0.6, 1.0),
    Vec2::new(0.4, 1.0),
    Vec2::new(0.4, 0.5),
    Vec2::new(0.4, 0.3),
    Vec2::new(0.6, 0.3),
    Vec2::new(0.6, 0.5),
    Vec2::new(0.4, 0.5),
    Vec2::new(0.4, 1.0),
];

const SLASH: &[Vec2] = &[
    Vec2::new(0.0, 1.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(0.0, 1.0),
];

const HEART: &[Vec2] = &[
    Vec2::new(0.5, 1.0),
    Vec2::new(0.0, 0.5),
    Vec2::new(0.0, CORNER),
    Vec2::new(CORNER, 0.0),
    Vec2::new(0.5, CORNER),
    Vec2::new(1.0 - CORNER, 0.0),
    Vec2::new(1.0, CORNER),
    Vec2::new(1.0, 0.5),
    Vec2::new(0.5, 1.0),
];

const STAR: &[Vec2] = &[
    Vec2::new(0.2, 1.0),
    Vec2::new(0.5, 0.0),
    Vec2::new(0.8, 1.0),
    Vec2::new(0.0, 0.4),
    Vec2::new(1.0, 0.4),
    Vec2::new(0.2, 1.0),
];

pub fn char_shape(ch: char) -> &'static [Vec2] {
    match ch {
        ' ' => SPACE,
        'A' => UPPER_A,
        'B' => UPPER_B,
        'C' => UPPER_C,
        'D' => UPPER_D,
        'E' => UPPER_E,
        'F' => UPPER_F,
        'G' => UPPER_G,
        'H' => UPPER_H,
        'I' => UPPER_I,
        'J' => UPPER_J,
        'K' => UPPER_K,
        'L' => UPPER_L,
        'M' => UPPER_M,
        'N' => UPPER_N,
        'O' => UPPER_O,
        'P' => UPPER_P,
        'Q' => UPPER_Q,
        'R' => UPPER_R,
        'S' => UPPER_S,
        'T' => UPPER_T,
        'U' => UPPER_U,
        'V' => UPPER_V,
        'W' => UPPER_W,
        'X' => UPPER_X,
        'Y' => UPPER_Y,
        'Z' => UPPER_Z,
        '0' => DIGIT_0,
        '1' => DIGIT_1,
        '2' => DIGIT_2,
        '3' => DIGIT_3,
        '4' => DIGIT_4,
        '5' => DIGIT_5,
        '6' => DIGIT_6,
        '7' => DIGIT_7,
        '8' => DIGIT_8,
        '9' => DIGIT_9,
        ':' => COLON,
        '/' => SLASH,
        '.' => PERIOD,
        'h' => HEART,
        's' => STAR,
        _ => UNKNOWN,
    }
}
