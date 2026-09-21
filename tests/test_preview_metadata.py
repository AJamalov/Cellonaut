from cellonaut.gui.preview_metadata import (
    is_overlay_preview_sidecar,
    parse_hex_color,
    parse_ome_display_sidecar,
)


def test_parse_hex_color_accepts_css_hex_and_fallback():
    assert parse_hex_color("#00FF80").tolist() == [0.0, 255.0, 128.0]
    assert parse_hex_color("not-a-color", fallback=(1, 2, 3)).tolist() == [1.0, 2.0, 3.0]


def test_overlay_sidecar_type_detection():
    assert is_overlay_preview_sidecar({"type": "cellonaut_overlay_preview"})
    assert is_overlay_preview_sidecar({"type": "Cellonaut_Cellpose_Outline_Stack"})
    assert not is_overlay_preview_sidecar({"type": "other"})


def test_parse_ome_display_sidecar_reads_channel_names_and_colors():
    ome_xml = """
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
      <Image>
        <Pixels>
          <Channel Name="GFP" Color="65280" />
          <Channel Name="mCherry" Color="16711680" />
        </Pixels>
      </Image>
    </OME>
    """

    sidecar = parse_ome_display_sidecar(ome_xml)

    assert sidecar["layer_labels"] == ["GFP", "mCherry"]
    assert sidecar["layer_colors"] == ["#00FF00", "#FF0000"]


def test_parse_ome_display_sidecar_reads_24bit_channel_colors():
    ome_xml = """
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
      <Image>
        <Pixels>
          <Channel Name="GFP" Color="65280" />
          <Channel Name="TxRed" Color="16711935" />
          <Channel Name="DAPI" Color="65535" />
        </Pixels>
      </Image>
    </OME>
    """

    sidecar = parse_ome_display_sidecar(ome_xml)

    assert sidecar["layer_colors"] == ["#00FF00", "#FF00FF", "#00FFFF"]


def test_parse_ome_display_sidecar_rejects_entity_expansion():
    ome_xml = '<!DOCTYPE OME [<!ENTITY payload "unsafe">]><OME><Channel Name="&payload;" /></OME>'

    assert parse_ome_display_sidecar(ome_xml) == {}
