use bevy::math::Vec2;

#[derive(Clone, Copy, Debug, Default)]
pub struct Circle {
    pub centre: Vec2,
    pub radius: f32,
}

impl Circle {
    pub fn contains(&self, v: Vec2) -> bool {
        v.distance(self.centre) < self.radius
    }
}

#[derive(Clone, Copy, Debug, Default)]
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

    pub fn grow(&self, by: f32) -> Self {
        let step = self.delta().normalize() * by;
        Self {
            start: self.start - step,
            end: self.end + step,
        }
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

    pub fn overlaps_with_circle(&self, circle: &Circle) -> bool {
        if circle.contains(self.start) || circle.contains(self.end) {
            return true;
        }
        let to_edge = self.delta().normalize() * circle.radius;
        let perp = Vec2 {
            x: to_edge.y,
            y: -to_edge.x,
        };
        Self {
            start: circle.centre,
            end: circle.centre + perp,
        }
        .intersect(self)
        .is_some()
            || Self {
                start: circle.centre,
                end: circle.centre - perp,
            }
            .intersect(self)
            .is_some()
    }
}
