#!/usr/bin/env python3
"""Create an animated 3D globe showing debris orbits over time.

Reads `debris_catalog_cleaned.csv` (must include `line1` and `line2`) and
generates `debris_3d_anim.html` with Play/Pause controls.
"""
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from skyfield.api import Loader, EarthSatellite


EARTH_RADIUS_KM = 6371.0088


def make_earth_surface(res=80, radius=EARTH_RADIUS_KM):
    u = np.linspace(0, 2 * np.pi, res)
    v = np.linspace(-np.pi / 2, np.pi / 2, res // 2)
    u, v = np.meshgrid(u, v)
    x = radius * np.cos(v) * np.cos(u)
    y = radius * np.cos(v) * np.sin(u)
    z = radius * np.sin(v)
    return x, y, z


def compute_positions_from_tles(df, ts, duration_minutes=90, frames=60):
    # Build time array
    now = datetime.utcnow()
    offsets = np.linspace(0, duration_minutes * 60, frames)
    times = ts.utc(now.year, now.month, now.day, now.hour, now.minute, now.second + offsets)

    sats = []
    for _, r in df.iterrows():
        if 'line1' in df.columns and 'line2' in df.columns and not pd.isna(r['line1']) and not pd.isna(r['line2']):
            try:
                sat = EarthSatellite(r['line1'], r['line2'], r.get('name', ''), ts)
                sats.append(sat)
            except Exception:
                sats.append(None)
        else:
            sats.append(None)

    num_sats = len(sats)
    xyz = np.zeros((num_sats, 3, frames))
    for i, sat in enumerate(sats):
        if sat is None:
            xyz[i, :, :] = np.nan
            continue
        try:
            geoc = sat.at(times)
            pos = geoc.position.km  # (3, frames)
            xyz[i, :, :] = pos
        except Exception:
            xyz[i, :, :] = np.nan

    return times, xyz


def build_figure(df, xyz, times, out_html='debris_3d_anim.html'):
    # Prepare marker sizes/colors from initial altitudes
    init_r = np.linalg.norm(xyz[:, :, 0], axis=1)
    alt0 = init_r - EARTH_RADIUS_KM
    sizes = np.clip(12 * (1.0 / (np.sqrt(alt0 + 1) / 5.0)), 3, 14)

    # Earth surface
    ex, ey, ez = make_earth_surface(res=120)
    surface = go.Surface(x=ex, y=ey, z=ez, surfacecolor=ez, showscale=False, colorscale='Blues', opacity=0.95)

    # Build frames
    frames = []
    num_frames = xyz.shape[2]
    for f in range(num_frames):
        xs = xyz[:, 0, f]
        ys = xyz[:, 1, f]
        zs = xyz[:, 2, f]
        # compute altitude colors
        rs = np.sqrt(xs * xs + ys * ys + zs * zs)
        alt_km = rs - EARTH_RADIUS_KM
        marker = dict(size=sizes, color=alt_km, colorscale='Viridis', cmin=0, cmax=2000, opacity=0.9)
        scatter = go.Scatter3d(x=xs, y=ys, z=zs, mode='markers', marker=marker,
                               text=df.get('name', df.index.astype(str)), hoverinfo='text')
        frames.append(go.Frame(data=[surface, scatter], name=str(f)))

    # Initial data (frame 0)
    init_x = xyz[:, 0, 0]
    init_y = xyz[:, 1, 0]
    init_z = xyz[:, 2, 0]
    init_r = np.sqrt(init_x * init_x + init_y * init_y + init_z * init_z)
    init_alt = init_r - EARTH_RADIUS_KM
    init_marker = dict(size=sizes, color=init_alt, colorscale='Viridis', cmin=0, cmax=2000, opacity=0.9)
    scatter0 = go.Scatter3d(x=init_x, y=init_y, z=init_z, mode='markers', marker=init_marker,
                            text=df.get('name', df.index.astype(str)), hoverinfo='text')

    layout = go.Layout(
        paper_bgcolor='black', plot_bgcolor='black',
        scene=dict(aspectmode='data', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   camera=dict(eye=dict(x=1.7, y=1.2, z=0.8))),
        updatemenus=[dict(type='buttons', showactive=False,
                          y=1, x=0.8, xanchor='left', yanchor='top',
                          buttons=[dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=100, redraw=True),
                                                                                             fromcurrent=True, mode='immediate')]),
                                   dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                                                                 mode='immediate', transition=dict(duration=0))])])],
        title=dict(text='Space Debris Orbits', x=0.5, xanchor='center', font=dict(color='white')),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig = go.Figure(data=[surface, scatter0], layout=layout, frames=frames)
    # build slider
    sliders = [dict(steps=[dict(method='animate', args=[[str(k)], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))], label=str(k)) for k in range(len(frames))],
                    active=0, x=0.1, y=0, xanchor='left', yanchor='top')]
    fig.update_layout(sliders=sliders)

    fig.write_html(out_html, auto_open=False)
    print(f'Wrote {out_html} ({len(frames)} frames)')


def main(csv_path='debris_catalog_cleaned.csv', out_html='debris_3d_anim.html', duration_minutes=90, frames=60):
    if not os.path.exists(csv_path):
        print(f'CSV not found: {csv_path}', file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if 'line1' not in df.columns or 'line2' not in df.columns:
        print('CSV must contain `line1` and `line2` columns for TLE propagation.', file=sys.stderr)
        sys.exit(1)

    load = Loader('./.skyfield-cache')
    ts = load.timescale()

    times, xyz = compute_positions_from_tles(df, ts, duration_minutes=duration_minutes, frames=frames)

    build_figure(df, xyz, times, out_html=out_html)


if __name__ == '__main__':
    main()
