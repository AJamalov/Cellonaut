"""Nested combined masks must load and use an image-backed dependency."""
from types import SimpleNamespace as NS
from typing import Any

import pytest

from cellonaut.masks import roi_processing as rois
from cellonaut.pipeline import planning
from cellonaut.pipeline.models import ImageDef, MeasurementTarget
from cellonaut.pipeline.roi_defs import mask_reference_image_keys


class Roi:
    def clone(self):
        return Roi()


@pytest.mark.parametrize('reuse', [False, True])
def test_nested_combined_processing_loads_and_uses_leaf_reference(tmp_path, monkeypatch, reuse):
    a = ImageDef('a', 'A', 'A', tmp_path / 'a.model')
    b = ImageDef('b', 'B', 'B', tmp_path / 'b.model')
    c = ImageDef('c', 'C', 'C', None, combined_mask_source_keys=['a', 'b'])
    d = ImageDef('d', 'D', 'D', None, combined_mask_source_keys=['a', 'b'])
    final = ImageDef('final', 'Final', 'Final', None, combined_mask_source_keys=['c', 'd'])
    final.mask_processing_steps = [{'type': 'translate', 'enabled': True, 'params': {'dx': 1}}]
    measured = ImageDef('measured', 'Measured', 'Measured', None)
    cfg: Any = NS(images=[a, b, c, d, final, measured], output_dir=tmp_path, mask_source_dir=None,
             reuse_existing_masks=reuse, probability_class_index=1, threshold_method='Default')
    monkeypatch.setattr(planning, 'existing_weka_threshold_mask_status', lambda *_: {'a': True, 'b': True})
    target = MeasurementTarget(source_image_key='measured', overlay_roi_keys=['final'])
    required = [a, b, c, d, final]
    needed = planning.image_keys_needed_for_processing(cfg, [target], required, 'sample')
    assert needed == {'measured', 'a', 'b'}
    images = {key: object() for key in needed}
    monkeypatch.setattr(rois, 'load_existing_class_rois', lambda *_args, **_kwargs: {1: Roi()})
    monkeypatch.setattr(rois, '_generate_missing_class_rois',
                        lambda **kwargs: (None, {index: Roi() for index in kwargs['missing_classes']}, False))
    monkeypatch.setattr(rois, 'union_shape_rois', lambda *_: Roi())
    references = []
    translated = Roi()
    monkeypatch.setattr(rois, 'apply_fiji_translate_step',
                        lambda roi, image, params: references.append(image) or translated)
    result, _ = rois.prepare_rois_for_defs(
        image_map=images, cfg=cfg, out_path=tmp_path / 'Masks', result_id='sample',
        roi_defs=required, segs={}, log_func=lambda _: None)
    assert references == [images['a']]
    assert result['final'] is translated


def test_reference_resolution_handles_class_keys_and_rejects_cycles():
    a = NS(key='a', combined_mask_source_keys=[])
    combined = NS(key='combined', combined_mask_source_keys=['a__class2', 'a'])
    assert mask_reference_image_keys(['combined'], [a, combined]) == ['a']
    a.combined_mask_source_keys = ['combined']
    with pytest.raises(ValueError, match='dependency cycle'):
        mask_reference_image_keys(['combined'], [a, combined])
