//! Small utilities shared by the feature modules (`madmom/utils/__init__.py`).

/// How two adjacent events are merged ([`combine_events`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Combine {
    /// Replace by the mean of the two events.
    Mean,
    /// Keep the earlier event.
    Left,
    /// Keep the later event.
    Right,
}

/// Combine all events that lie within `delta` of each other.
///
/// The reference walks the events once, comparing every event to the one it currently
/// holds; the 1e-12 added to `delta` is part of its behaviour (it makes a "just barely
/// equal" gap count as inside), so it is kept here rather than "cleaned up".
pub fn combine_events(events: &[f64], delta: f64, combine: Combine) -> Vec<f64> {
    if events.len() <= 1 {
        return events.to_vec();
    }
    let delta = delta + 1e-12;
    let mut out: Vec<f64> = Vec::with_capacity(events.len());
    let mut left = events[0];
    for right in &events[1..] {
        if *right - left <= delta {
            // the merged value belongs to the slot the reference is currently holding
            // (`events[idx]`), which is only written out when the next gap is too big
            left = match combine {
                Combine::Mean => 0.5 * (left + *right),
                Combine::Left => left,
                Combine::Right => *right,
            };
        } else {
            out.push(left);
            left = *right;
        }
    }
    out.push(left);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_within_delta_are_merged() {
        // the reference's own example shape: a cluster of onsets becomes one
        let events = [0.1, 0.12, 0.2, 0.5, 0.51, 1.0];
        assert_eq!(
            combine_events(&events, 0.02, Combine::Left),
            vec![0.1, 0.2, 0.5, 1.0]
        );
        assert_eq!(
            combine_events(&events, 0.02, Combine::Right),
            vec![0.12, 0.2, 0.51, 1.0]
        );
        let mean = combine_events(&events, 0.02, Combine::Mean);
        assert!((mean[0] - 0.11).abs() < 1e-12);
        assert!((mean[2] - 0.505).abs() < 1e-12);
    }

    #[test]
    fn fewer_than_two_events_come_back_untouched() {
        assert!(combine_events(&[], 0.03, Combine::Left).is_empty());
        assert_eq!(combine_events(&[1.5], 0.03, Combine::Mean), vec![1.5]);
    }
}
