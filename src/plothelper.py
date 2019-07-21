import matplotlib.lines
import cartopy.feature

legend_elements = [matplotlib.lines.Line2D([0], [0], marker='o', color='w', label='1000 MW', markerfacecolor='0', markersize=(1000/10)**0.5),
                   matplotlib.lines.Line2D([0], [0], marker='o', color='w', label='200 MW', markerfacecolor='0', markersize=(200/10)**0.5),
                   matplotlib.lines.Line2D([0], [0], marker='o', color='w', label='in', markerfacecolor='#6fd08c', markersize=(1000/10)**0.5),
                   matplotlib.lines.Line2D([0], [0], marker='o', color='w', label='out', markerfacecolor='#7b9ea8', markersize=(1000/10)**0.5)]


def add_water_to_axis(axis, geomap_scale):
    axis.add_feature(cartopy.feature.LAKES.with_scale(geomap_scale), facecolor="#f0f5ff", edgecolor="#000000", linewidth=.2, zorder=.5)
    axis.add_feature(cartopy.feature.LAND.with_scale(geomap_scale), facecolor="#ffffff", edgecolor="#000000", linewidth=.2)
    axis.background_patch.set_facecolor("#f0f5ff")

solar_color = "#e2b541"
wind_color = "#416ee2"

injection_positive_color = '#6fd08c'
injection_negative_color = '#7b9ea8'
