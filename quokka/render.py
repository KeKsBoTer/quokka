from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
import numpy as np
from . import quokka

@dataclass
class DVRSettings:
    enabled: bool = False,
    distance_scale: float = 1.0
    vmin: Optional[float] = None
    vmax: Optional[float] = None

    def _to_rust(self):
        return quokka.DVRSettings(
            self.enabled,
            self.distance_scale,
            self.vmin,
            self.vmax,
        )
    
    def from_dict(values:dict)->"DVRSettings":
        return DVRSettings(
            values["enabled"],
            values["distance_scale"],
            values["vmin"],
            values["vmax"]
        )

@dataclass
class IsoSettings:
    enabled: bool = True
    use_cube_surface_grad: bool = False

    shininess: float = 20.

    threshold: float = 0.5

    ambient_color: Tuple[float,float,float] = (0.,0.,0.)

    specular_color: Tuple[float,float,float] = (0.7,0.7,0.7)

    light_color:  Tuple[float,float,float] = (1.,1.,1.)

    diffuse_color: Tuple[float,float,float,float] = (1.0, 0.871, 0.671, 1.0)

    def _to_rust(self):
        return quokka.IsoSettings(
            self.enabled,
            self.use_cube_surface_grad,
            self.shininess,
            self.threshold,
            self.ambient_color,
            self.specular_color,
            self.light_color,
            self.diffuse_color
        )
    
    def from_dict(values:dict)->"IsoSettings":
        return IsoSettings(
            values["enabled"],
            values["use_cube_surface_grad"],
            values["shininess"],
            values["threshold"],
            _dict_to_tuple(values["ambient_color"]),
            _dict_to_tuple(values["specular_color"]),
            _dict_to_tuple(values["light_color"]),
            _dict_to_tuple(values["diffuse_color"]),
        )

@dataclass
class SSAOSettings:
    enabled: bool = True
    radius: float = 0.4
    bias: float = 0.02
    kernel_size: int = 64

    def _to_rust(self):
        return quokka.SSAOSettings(
            self.enabled,
            self.radius,
            self.bias,
            self.kernel_size
        )

    def from_dict(values:dict)->"SSAOSettings":
        return SSAOSettings(
            values["enabled"],
            values["radius"],
            values["bias"],
            values["kernel_size"]
        )


@dataclass
class RenderSettings:  
    spatial_filter: str = "linear"

    temporal_filter: str = "linear"

    dvr: DVRSettings = field(default_factory=lambda: DVRSettings())
    iso_surface: IsoSettings = field(default_factory=lambda: IsoSettings())
    ssao: SSAOSettings = field(default_factory=lambda: SSAOSettings())

    background_color: Tuple[float,float,float,float] = (0.,0.,0.,1.)

    near_clip_plane: Optional[float] = None

    def _to_rust(self):
        return quokka.RenderSettings(
            self.spatial_filter,
            self.temporal_filter,
            self.dvr._to_rust(),
            self.iso_surface._to_rust(),
            self.ssao._to_rust(),
            self.background_color,
            self.near_clip_plane
        )
    
    def from_dict(values:dict)->"RenderSettings":
        return RenderSettings(
            values["spatial_filter"],
            values["temporal_filter"],
            DVRSettings.from_dict(values["dvr"]),
            IsoSettings.from_dict(values["iso_surface"]),
            SSAOSettings.from_dict(values["ssao"]),
            _dict_to_tuple(values["background_color"]),
            values.get("near_clip_plane",None)
        )
@dataclass
class Preset:
    name: str
    render_settings: RenderSettings
    cmap: Optional[dict[str,list[float]]] = None
    camera: Optional[Tuple[float,float,float,float]] = None

    def _to_rust(self):
        cmap = None
        if self.cmap is not None:
            cmap = quokka.LinearSegmentedColorMap(self.cmap["r"],self.cmap["g"],self.cmap["b"],self.cmap["a"])
        return quokka.Preset(
            self.name,
            self.render_settings._to_rust(),
            cmap,
            self.camera
        )
    
    def from_dict(values:dict)->"Preset":

        cmap = values.get("cmap",None)
        if cmap is not None:
            for f in cmap.keys():   
                cmap[f] = [tuple(c) for c in cmap[f]]

        camera = values.get("camera",None)
        if camera is not None:
            camera = _dict_to_tuple(camera)

        return Preset(
            values["name"],
            RenderSettings.from_dict(values["render_settings"]),
            cmap,    
            camera
        )


def render(
    volume: np.ndarray,
    preset: Preset,
    width: int = 1024,
    height: int = 1024,
) -> np.ndarray:
    """renders a single or multiple images of a volume

    Args:
        volume (np.ndarray): volume data of shape [D, H, W]
        cmap (Colormap): colormap to use for rendering. Defaults to matplotlib's default colormap.
        preset (quokka.Preset): rendering preset
        width (int, optional): image width. Defaults to 1024.
        height (int, optional): image height. Defaults to 1024.

    Returns:
        np.ndarray: [H, W, 4]
    """

    if volume.ndim != 3:
        raise ValueError(
            "volume must have shape  [D,H,W] "
        )

    
    frames = quokka.render_video(
        np.ascontiguousarray(volume).astype(np.float16),
        width,
        height,
        [0.],
        preset._to_rust()
    )
    return frames[0]


def _dict_to_tuple(d:dict)->tuple:
    try:
        fields =[d["x"],d["y"],d["z"]]
        if "w" in d:
            fields.append(d["w"])
        return tuple(fields)
    except KeyError:
        fields =[d["r"],d["g"],d["b"]]
        if "a" in d:
            fields.append(d["a"])
        return tuple(fields)