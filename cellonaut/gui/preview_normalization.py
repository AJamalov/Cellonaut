"""Normalize microscopy image axes into one model used by the preview UI."""

from __future__ import annotations

import numpy as np

from cellonaut.gui.preview_metadata import is_overlay_preview_sidecar


class CellonautGuiPreviewNormalizationMixin:
    """Arrange preview arrays as time, Z, channel, height, and width."""

    # Prefer axes before height/width, but also consider a trailing color axis
    # of size 2, 3, or 4. Shape alone can be ambiguous.
    def _pick_channel_axis_for_stack(
        self,
        arr: np.ndarray,
        channel_count: int | None = None,
    ) -> int | None:
        arr = np.asarray(arr)
        if arr.ndim < 3:
            return None

        candidate_axes = list(range(max(0, arr.ndim - 2)))
        if arr.shape[-1] in (2, 3, 4):
            candidate_axes.append(arr.ndim - 1)

        if channel_count is not None and int(channel_count) > 1:
            matching_axes = [i for i in candidate_axes if int(arr.shape[i]) == int(channel_count)]
            if matching_axes:
                if arr.shape[-1] in (2, 3, 4) and arr.ndim - 1 in matching_axes:
                    return arr.ndim - 1
                return matching_axes[0]

        candidates = [i for i in range(arr.ndim - 2) if int(arr.shape[i]) <= 16]
        if not candidates:
            return None
        if arr.shape[-1] in (2, 3, 4):
            return arr.ndim - 1
        return candidates[0]

    def normalize_tiff_to_tzcyx(
        self,
        arr: np.ndarray,
        sidecar: dict | None = None,
    ) -> dict:
        """Return a display model whose data has axes (T, Z, C, Y, X).

        Valid metadata wins; otherwise shape/layer-count heuristics infer axes.
        Missing dimensions become singleton axes. Unsupported metadata axes use
        index zero; RGB alpha is discarded. Values are not intensity-normalized
        here, and returned arrays may share storage with the input.
        """
        sidecar = sidecar or {}
        is_overlay = is_overlay_preview_sidecar(sidecar)
        axes = str(sidecar.get("axes", "") or "").strip().upper()
        arr_raw = np.asarray(arr)

        axes_model = self._normalize_tiff_with_axes(arr_raw, axes, is_overlay)
        if axes_model is not None:
            return axes_model

        # Keep height and width, even for single-row or single-column images.
        singleton_axes = tuple(i for i, size in enumerate(arr_raw.shape[:-2]) if size == 1)
        arr = np.squeeze(arr_raw, axis=singleton_axes)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)

        sidecar_labels = list(sidecar.get("layer_labels", []) or [])
        sidecar_channel_count = len(sidecar_labels) if sidecar_labels else None

        if arr.ndim == 2:
            data = arr[np.newaxis, np.newaxis, np.newaxis, :, :]
            return self._normalized_preview_model(data, arr, is_overlay)

        if arr.ndim == 3:
            if is_overlay:
                channel_axis = self._pick_channel_axis_for_stack(arr, channel_count=sidecar_channel_count)
                arr_cyx = np.moveaxis(arr, 0 if channel_axis is None else channel_axis, 0)
                return self._normalized_preview_model(
                    arr_cyx[np.newaxis, np.newaxis, :, :, :],
                    arr,
                    True,
                )

            if arr.shape[-1] in (3, 4):
                arr_cyx = np.moveaxis(arr[..., :3], -1, 0)
                return self._normalized_preview_model(
                    arr_cyx[np.newaxis, np.newaxis, :, :, :],
                    arr,
                    False,
                )

            channel_axis = self._pick_channel_axis_for_stack(arr)
            if channel_axis is not None:
                arr_cyx = np.moveaxis(arr, channel_axis, 0)
                return self._normalized_preview_model(
                    arr_cyx[np.newaxis, np.newaxis, :, :, :],
                    arr,
                    False,
                )

            return self._normalized_preview_model(
                arr[np.newaxis, :, np.newaxis, :, :],
                arr,
                False,
            )

        if arr.ndim == 4:
            if arr.shape[-1] in (3, 4):
                moved = np.moveaxis(arr[..., :3], -1, -3)
                data = moved[np.newaxis, ...]
                return self._normalized_preview_model(data, arr, False)

            small_axes = [i for i, size in enumerate(arr.shape[:-2]) if int(size) <= 16]
            if small_axes:
                moved = np.moveaxis(arr, small_axes[-1], -3)
                if moved.ndim == 4:
                    data = moved[np.newaxis, ...]
                    if data.ndim == 5:
                        return self._normalized_preview_model(data, arr, is_overlay)

            y, x = arr.shape[-2:]
            page_count = int(np.prod(arr.shape[:-2]))
            data = arr.reshape(page_count, y, x).reshape(1, page_count, 1, y, x)
            return self._normalized_preview_model(data, arr, is_overlay)

        if arr.ndim >= 5:
            y, x = arr.shape[-2:]
            prefix_shape = arr.shape[:-2]
            channel_axis = next(
                (i for i, size in enumerate(prefix_shape) if int(size) <= 16),
                None,
            )

            if channel_axis is not None:
                moved = np.moveaxis(arr, channel_axis, -3)
                prefix = moved.shape[:-3]
                c, y, x = moved.shape[-3:]
                flat_pages = int(np.prod(prefix)) if prefix else 1
                data = moved.reshape(flat_pages, c, y, x).reshape(1, flat_pages, c, y, x)
            else:
                flat_pages = int(np.prod(prefix_shape)) if prefix_shape else 1
                data = arr.reshape(flat_pages, y, x).reshape(1, flat_pages, 1, y, x)

            return self._normalized_preview_model(data, arr, is_overlay)

        raise ValueError(f"Unsupported TIFF shape: {tuple(arr.shape)}")

    # Metadata wins over shape heuristics because T, Z, and C dimensions can all
    # have the same small sizes in microscopy stacks.
    def _normalize_tiff_with_axes(
        self,
        arr: np.ndarray,
        axes: str,
        is_overlay: bool,
    ) -> dict | None:
        if not axes or len(axes) != arr.ndim or "Y" not in axes or "X" not in axes:
            return None

        work = np.asarray(arr)
        dims = list(axes)
        if "Z" not in dims:
            for page_label in ("Q", "I"):
                if page_label in dims:
                    dims[dims.index(page_label)] = "Z"
                    break
        sample_axis = dims.index("S") if "S" in dims else None
        if "C" not in dims and sample_axis is not None and work.shape[sample_axis] in (3, 4):
            if work.shape[sample_axis] == 4:
                work = np.take(work, indices=range(3), axis=sample_axis)
            dims[sample_axis] = "C"

        for dim in list(dims):
            if dim in {"T", "Z", "C", "Y", "X"}:
                continue
            axis = dims.index(dim)
            work = np.take(work, indices=0, axis=axis)
            dims.pop(axis)

        target_dims = ["T", "Z", "C", "Y", "X"]
        ordered_existing = [dim for dim in target_dims if dim in dims]
        transpose_axes = [dims.index(dim) for dim in ordered_existing]
        work = np.transpose(work, transpose_axes) if transpose_axes else work
        dims = ordered_existing

        for index, dim in enumerate(target_dims):
            if dim not in dims:
                work = np.expand_dims(work, axis=index)
                dims.insert(index, dim)

        if work.ndim != 5:
            return None

        model = self._normalized_preview_model(work, arr, is_overlay)
        model["axes"] = axes
        model["uses_metadata_axes"] = True
        model["metadata_channel_axis"] = "C" in axes
        model["metadata_sample_axis"] = "S" in axes
        return model

    @staticmethod
    def _normalized_preview_model(data: np.ndarray, source: np.ndarray, is_overlay: bool) -> dict:
        return {
            "data": data,
            "is_overlay": is_overlay,
            "source_shape": tuple(source.shape),
            "dtype": str(source.dtype),
        }

    # ND2 readers provide ordered dimension names; preserve that order instead
    # of applying TIFF shape heuristics whenever it matches the returned array.
    def normalize_nd2_to_tzcyx(self, arr: np.ndarray, sizes: dict) -> dict:
        """Arrange ND2 pixels for display using sizes' insertion-ordered axis names.

        Preserve T/Z/C; select index zero for other acquisition axes such as P.
        Missing/inconsistent axis metadata falls back to TIFF shape inference.
        The returned model has TZCYX data and source metadata, without scaling.
        """
        arr = np.asarray(arr)
        dim_order = [str(dim) for dim in dict(sizes or {})]

        if len(dim_order) != arr.ndim:
            model = self.normalize_tiff_to_tzcyx(arr)
            model["nd2_sizes"] = dict(sizes or {})
            return model

        work = arr
        dims = list(dim_order)
        for dim in list(dims):
            if dim in {"T", "Z", "C", "Y", "X"}:
                continue
            axis = dims.index(dim)
            work = np.take(work, indices=0, axis=axis)
            dims.pop(axis)

        if "Y" not in dims or "X" not in dims:
            model = self.normalize_tiff_to_tzcyx(work)
            model["nd2_sizes"] = dict(sizes or {})
            return model

        target_dims = ["T", "Z", "C", "Y", "X"]
        ordered_existing = [dim for dim in target_dims if dim in dims]
        axes = [dims.index(dim) for dim in ordered_existing]
        work = np.transpose(work, axes) if axes else work
        dims = ordered_existing

        for index, dim in enumerate(target_dims):
            if dim not in dims:
                work = np.expand_dims(work, axis=index)
                dims.insert(index, dim)

        if work.ndim != 5:
            raise ValueError(f"Unsupported ND2 shape after normalization: {tuple(work.shape)}")

        return {
            "data": work,
            "is_overlay": False,
            "source_shape": tuple(arr.shape),
            "dtype": str(arr.dtype),
            "nd2_sizes": dict(sizes or {}),
            "nd2_dim_order": dim_order,
        }
