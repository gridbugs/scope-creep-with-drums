use bevy::math::Vec2;

#[derive(Clone, Copy, Debug)]
pub struct Seg2 {
    pub start: Vec2,
    pub end: Vec2,
}

fn cross2(v: Vec2, w: Vec2) -> f32 {
    (v.x * w.y) - (v.y * w.x)
}

impl Seg2 {
    pub fn new(start: Vec2, end: Vec2) -> Self {
        Self { start, end }
    }

    pub fn delta(&self) -> Vec2 {
        self.end - self.start
    }

    pub fn crosses_x_axis_at(&self) -> Option<Vec2> {
        if (self.start.y >= 0. && self.end.y >= 0.) || (self.start.y <= 0. && self.end.y <= 0.) {
            return None;
        }
        let delta = self.delta();
        let mult = -self.start.y / delta.y;
        Some(self.start + (delta * mult))
    }

    pub fn intersect(&self, other: &Self) -> Option<Vec2> {
        let p = self.start;
        let r = self.delta();
        let q = other.start;
        let s = other.delta();
        let denom = cross2(r, s);
        let t_num = cross2(q - p, s);
        let u_num = cross2(q - p, r);
        if u_num == 0. && denom == 0. {
            // Line segments are colinear. A complete solution would handle this case but it's not
            // necessary for this application.
            return None;
        }
        if denom == 0. {
            // Line segments are parallel and non-intersecting.
            return None;
        }
        let t = t_num / denom;
        let u = u_num / denom;
        if (0. ..=1.).contains(&t) && (0. ..=1.).contains(&u) {
            Some(p + (t * r))
        } else {
            // Line segments are non-parallel but do not intersect
            None
        }
    }
}
