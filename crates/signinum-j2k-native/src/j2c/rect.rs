#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct IntRect {
    pub(crate) x0: u32,
    pub(crate) y0: u32,
    pub(crate) x1: u32,
    pub(crate) y1: u32,
}

impl IntRect {
    pub(crate) fn from_ltrb(x0: u32, y0: u32, x1: u32, y1: u32) -> Self {
        Self { x0, y0, x1, y1 }
    }

    pub(crate) fn from_xywh(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self {
            x0: x,
            y0: y,
            x1: x + w,
            y1: y + h,
        }
    }

    pub(crate) fn width(&self) -> u32 {
        // See B-11.
        self.x1 - self.x0
    }

    pub(crate) fn height(&self) -> u32 {
        // See B-11.
        self.y1 - self.y0
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }

    pub(crate) fn intersect(&self, other: Self) -> Self {
        if self.x1 < other.x0 || other.x1 < self.x0 || self.y1 < other.y0 || other.y1 < self.y0 {
            Self::from_xywh(0, 0, 0, 0)
        } else {
            Self::from_ltrb(
                u32::max(self.x0, other.x0),
                u32::max(self.y0, other.y0),
                u32::min(self.x1, other.x1),
                u32::min(self.y1, other.y1),
            )
        }
    }

    pub(crate) fn union(&self, other: Self) -> Self {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return *self;
        }
        Self::from_ltrb(
            self.x0.min(other.x0),
            self.y0.min(other.y0),
            self.x1.max(other.x1),
            self.y1.max(other.y1),
        )
    }

    pub(crate) fn expanded_within(&self, margin: u32, bounds: Self) -> Self {
        Self::from_ltrb(
            self.x0.saturating_sub(margin).max(bounds.x0),
            self.y0.saturating_sub(margin).max(bounds.y0),
            self.x1.saturating_add(margin).min(bounds.x1),
            self.y1.saturating_add(margin).min(bounds.y1),
        )
    }

    pub(crate) fn intersects(&self, other: Self) -> bool {
        self.x0 < other.x1 && other.x0 < self.x1 && self.y0 < other.y1 && other.y0 < self.y1
    }
}

impl From<IntRect> for crate::J2kRect {
    fn from(rect: IntRect) -> Self {
        Self {
            x0: rect.x0,
            y0: rect.y0,
            x1: rect.x1,
            y1: rect.y1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::J2kRect;

    #[test]
    fn int_rect_converts_to_external_j2k_rect() {
        let rect = J2kRect::from(IntRect::from_ltrb(3, 5, 11, 17));

        assert_eq!(rect.x0, 3);
        assert_eq!(rect.y0, 5);
        assert_eq!(rect.x1, 11);
        assert_eq!(rect.y1, 17);
    }
}
