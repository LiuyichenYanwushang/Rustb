use Rustb::model::{model_utils as nested_model_utils, utils as nested_utils};
use Rustb::{find_R, model_utils, remove_col, remove_row, utils};
use ndarray::array;

// Physics exposes inherent methods, so resolving these imports is the API check.
#[allow(unused_imports)]
use Rustb::{
    model::{model_physics as nested_model_physics, physics as nested_physics},
    model_physics, physics,
};

#[test]
fn model_module_paths_remain_compatible() {
    let values = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];

    for (find, drop_col, drop_row) in [
        (find_R, remove_col, remove_row),
        (utils::find_R, utils::remove_col, utils::remove_row),
        (
            model_utils::find_R,
            model_utils::remove_col,
            model_utils::remove_row,
        ),
        (
            nested_utils::find_R,
            nested_utils::remove_col,
            nested_utils::remove_row,
        ),
        (
            nested_model_utils::find_R,
            nested_model_utils::remove_col,
            nested_model_utils::remove_row,
        ),
    ] {
        assert_eq!(find(&values, &array![4, 5, 6]), Some(1));
        assert_eq!(find(&values, &array![0, 0, 0]), None);
        assert_eq!(drop_col(values.clone(), 1), array![[1, 3], [4, 6], [7, 9]]);
        assert_eq!(drop_row(values.clone(), 1), array![[1, 2, 3], [7, 8, 9]]);
    }
}
