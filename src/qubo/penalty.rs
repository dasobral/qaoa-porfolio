use super::QUBOMatrix;

pub struct PenaltyBuilder;

impl PenaltyBuilder {
    pub fn budget(q: &mut QUBOMatrix, target_k: usize, weight: f64) {
        let n = q.num_variables();
        let diagonal = weight * (1.0 - 2.0 * target_k as f64);
        let off_diagonal = 2.0 * weight;

        for i in 0..n {
            q.add(i, i, diagonal)
                .expect("budget penalty indices are within QUBO dimensions");
            for j in (i + 1)..n {
                q.add(i, j, off_diagonal)
                    .expect("budget penalty indices are within QUBO dimensions");
            }
        }

        q.add_offset(weight * (target_k * target_k) as f64)
            .expect("budget penalty offset is finite");
    }

    pub fn position_limit(
        q: &mut QUBOMatrix,
        group_indices: &[usize],
        max_from_group: usize,
        weight: f64,
    ) {
        let diagonal = weight * (1.0 - 2.0 * max_from_group as f64);
        let off_diagonal = 2.0 * weight;

        for (position, &i) in group_indices.iter().enumerate() {
            if i >= q.num_variables() {
                continue;
            }
            q.add(i, i, diagonal)
                .expect("position penalty value is finite");
            for &j in group_indices.iter().skip(position + 1) {
                if j < q.num_variables() {
                    q.add(i, j, off_diagonal)
                        .expect("position penalty value is finite");
                }
            }
        }

        q.add_offset(weight * (max_from_group * max_from_group) as f64)
            .expect("position penalty offset is finite");
    }

    pub fn diversity(
        q: &mut QUBOMatrix,
        class_groups: &[Vec<usize>],
        min_classes: usize,
        weight: f64,
    ) {
        if min_classes == 0 || class_groups.is_empty() {
            return;
        }

        let class_reward = weight / min_classes as f64;
        for group in class_groups {
            if group.is_empty() {
                continue;
            }
            let per_asset_reward = -class_reward / group.len() as f64;
            for &index in group {
                if index < q.num_variables() {
                    q.add(index, index, per_asset_reward)
                        .expect("diversity penalty value is finite");
                }
            }
            for (position, &i) in group.iter().enumerate() {
                if i >= q.num_variables() {
                    continue;
                }
                for &j in group.iter().skip(position + 1) {
                    if j < q.num_variables() {
                        q.add(i, j, weight)
                            .expect("diversity penalty value is finite");
                    }
                }
            }
        }
    }
}
