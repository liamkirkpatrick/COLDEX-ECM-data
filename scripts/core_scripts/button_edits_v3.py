#%% Import packages

# general
import os
import re
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import SpanSelector

# my own modules
import sys
sys.path.append("../core_scripts/")
from ECMclass import ECM

#%% Load data
path_to_input_data = '../../data/not-cleaned/'
path_to_output_data = '../../data/clean/'
path_to_metadata = '../../data/'
metadata_file = 'metadata.csv'
window = 10

meta = pd.read_csv(path_to_metadata + metadata_file)

def print_startup_intro():
    print(
        "\n"
        "============================================================\n"
        "ECM MANUAL EDITOR\n"
        "============================================================\n"
        "This tool loads not-cleaned ECM files and writes edited files\n"
        "to data/clean so original inputs remain untouched. Each run\n"
        "shows AC (left pair) and DC (right pair) side-by-side.\n"
        "\n"
        "Keyboard and interaction commands:\n"
        "  Drag on ANY subplot (no modifier): mark surface_imperfection (gray)\n"
        "  Hold 't' while dragging: mark tephra_or_layer (orange)\n"
        "  Hold 'a' while dragging: mark rock_or_debris (purple)\n"
        "  d : delete last selection for current Y\n"
        "  n : next Y-dimension\n"
        "  b : previous Y-dimension\n"
        "  w : save current file and continue\n"
        "  r : skip current file without saving\n"
        "  q : quit\n"
        "============================================================\n"
    )


def extract_leading_section(section_value):
    match = re.match(r'^\s*(\d+)', str(section_value))
    if match is None:
        return None
    return int(match.group(1))


def normalize_section_for_lookup(section_value):
    return str(section_value).strip()


def format_note_title(note, wrap_interval=30):
    note_text = str(note).strip()
    text = f"Note: {note_text}" if note_text else "Note:"
    parts = []
    start = 0
    while start + wrap_interval < len(text):
        split_at = text.find(' ', start + wrap_interval)
        if split_at == -1:
            break
        parts.append(text[start:split_at])
        start = split_at + 1
    parts.append(text[start:])
    return "\n".join(parts)


def prompt_for_core(core_values):
    while True:
        core_id = input("Enter core ID: ").strip()
        if core_id in core_values:
            return core_id
        print("Core ID not found in metadata. Please enter a valid core ID.")


def prompt_for_section(prompt):
    while True:
        value = input(prompt).strip()
        if re.fullmatch(r'\d{1,4}', value):
            return int(value)
        print("Please enter a 1-4 digit integer.")


core_values = set(meta['core'].dropna().astype(str).unique())
print_startup_intro()
selected_core = prompt_for_core(core_values)

start_section = prompt_for_section("Enter starting section (1-4 digit integer): ")
while True:
    end_section = prompt_for_section("Enter maximum section (1-4 digit integer): ")
    if start_section < end_section:
        break
    print("Maximum section must be greater than starting section.")

meta_core = meta[meta['core'].astype(str) == selected_core].copy()
meta_core['section_num'] = meta_core['section'].apply(extract_leading_section)
meta_core = meta_core[meta_core['section_num'].notna()]
meta_core = meta_core[
    (meta_core['section_num'] >= start_section) &
    (meta_core['section_num'] <= end_section)
]

if meta_core.empty:
    print("No metadata rows matched the selected core and section range.")
    sys.exit(1)

data = [
    {
        'ecm': ECM(row['core'], row['section'], row['face'], row['ACorDC'], path_to_input_data, path_to_metadata + metadata_file),
        'note': row['note'] if pd.notna(row.get('note')) else '',
        'core': row['core'],
        'section': row['section'],
        'face': row['face'],
        'acordc': row['ACorDC']
    }
    for _, row in meta_core.iterrows()
]

entry_lookup = {
    (
        entry['core'],
        normalize_section_for_lookup(entry['section']),
        entry['face'],
        entry['acordc']
    ): entry
    for entry in data
}

# import colormaps
cbar_2d = matplotlib.colormaps['Spectral']
cbar_line = matplotlib.colormaps['coolwarm']


#%% Define processing function
def process_ecm(entry, entry_number, total_entries):
    d = entry['ecm']
    note = entry['note']
    print(f"Plotting {d.core}, section {d.section}-{d.face}-{d.ACorDC}")


    # Extract raw vectors
    depth = d.depth
    y = d.y
    meas = d.meas
    if d.button_raw is not None:
        button = d.button_raw
    else:
        button = d.button
    
    d.smooth(window)
    depth_s = d.depth_s
    y_s = d.y_s
    meas_s = d.meas_s
    button_s = d.button_s

    # Initialize editable classification columns
    surface_imperfection = np.zeros(len(d.button), dtype=int)
    tephra_or_layer = np.zeros(len(d.button), dtype=int)
    rock_or_debris = np.zeros(len(d.button), dtype=int)


    # Build DataFrame
    df = pd.DataFrame({
        'True_depth(m)': depth,
        'Y_dimension(mm)': y,
        'meas': meas,
        'button': button,
        'surface_imperfection': surface_imperfection,
        'tephra_or_layer': tephra_or_layer,
        'rock_or_debris': rock_or_debris
    })

    df_s = pd.DataFrame({
        'True_depth(m)': depth_s,
        'Y_dimension(mm)': y_s,
        'meas': meas_s,
        'button': button_s,
    })

    # Filename for saving
    fname = f"{d.core}-{d.section}-{d.face}-{d.ACorDC}.csv"

    # Unique Y-dimension values
    y_values = sorted(df['Y_dimension(mm)'].unique())
    opposite_type = 'DC' if d.ACorDC == 'AC' else 'AC'
    opposite_entry = entry_lookup.get(
        (
            d.core,
            normalize_section_for_lookup(d.section),
            d.face,
            opposite_type
        )
    )

    fig = plt.figure(figsize=(18, 7))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.1, 1.1, 0.2, 1.1, 1.1], wspace=0.05)
    ax_ac_left = fig.add_subplot(gs[0, 0])
    ax_ac_right = fig.add_subplot(gs[0, 1], sharey=ax_ac_left)
    ax_gap = fig.add_subplot(gs[0, 2])
    ax_dc_left = fig.add_subplot(gs[0, 3], sharey=ax_ac_left)
    ax_dc_right = fig.add_subplot(gs[0, 4], sharey=ax_ac_left)
    ax_gap.axis('off')

    pair_axes = {
        'AC': (ax_ac_left, ax_ac_right),
        'DC': (ax_dc_left, ax_dc_right),
    }
    all_data_axes = [ax_ac_left, ax_ac_right, ax_dc_left, ax_dc_right]

    ac_entry = entry if d.ACorDC == 'AC' else opposite_entry
    dc_entry = entry if d.ACorDC == 'DC' else opposite_entry

    def plot_side(side_entry, side_name):
        ax_left, ax_right = pair_axes[side_name]
        ax_left.set_xlabel('Distance Accross Core (mm)')
        ax_left.set_ylabel('True_depth(m)')
        ax_right.invert_yaxis()
        ax_right.set_ylabel('')
        ax_right.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax_right.set_xlabel('meas')

        if side_entry is None:
            ax_left.set_title(f"{side_name}\nMissing data")
            ax_right.set_title(f"{side_name} profiles")
            return None, {}

        side_d = side_entry['ecm']
        side_note = side_entry['note']
        side_d.smooth(window)
        side_depth_s = side_d.depth_s
        side_y_s = side_d.y_s
        side_meas_s = side_d.meas_s
        side_button_s = side_d.button_s

        ax_left.set_title(f"{side_name}\n{format_note_title(side_note)}")

        side_y_values = sorted(np.unique(side_y_s))
        lines = {}
        for idx, yv in enumerate(side_y_values):
            subset_mask = side_y_s == yv
            line, = ax_right.plot(
                side_meas_s[subset_mask],
                side_depth_s[subset_mask],
                color=cbar_line(idx / max(1, len(side_y_values))),
                label=str(yv)
            )
            lines[yv] = line

        ax_right.legend(
            title='Y_dimension(mm)',
            bbox_to_anchor=(0.5, -0.14),
            loc='upper center',
            ncol=3,
            fontsize='small'
        )

        # Plot button points
        side_df_s = pd.DataFrame({
            'True_depth(m)': side_depth_s,
            'Y_dimension(mm)': side_y_s,
            'meas': side_meas_s,
            'button': side_button_s,
        })
        button_mask = side_df_s['button'] == 0
        side_df_s.loc[button_mask, 'meas'] = np.nan
        for yv in side_y_values:
            subset = side_df_s[side_df_s['Y_dimension(mm)'] == yv]
            if not subset['meas'].isna().all():
                ax_right.plot(subset['meas'], subset['True_depth(m)'], color='black')

        # Plot top-down map
        x_unique = np.unique(side_y_s)
        y_unique = np.unique(side_depth_s)
        X, Y = np.meshgrid(x_unique, y_unique)
        Z = np.full(X.shape, np.nan)
        map_button_mask = np.full(X.shape, False)
        for i in range(len(side_meas_s)):
            ix = np.where(x_unique == side_y_s[i])[0]
            iy = np.where(y_unique == side_depth_s[i])[0]
            Z[iy, ix] = side_meas_s[i]
            if side_button_s[i] == 1:
                map_button_mask[iy, ix] = True

        ax_left.pcolormesh(X, Y, Z, shading='auto', cmap=cbar_2d)
        ax_left.pcolormesh(
            X,
            Y,
            np.ma.masked_where(~map_button_mask, Z),
            shading='auto',
            cmap=plt.cm.colors.ListedColormap(['black']),
            alpha=0.5
        )
        return side_depth_s, lines

    ac_depth_s, ac_lines = plot_side(ac_entry, 'AC')
    dc_depth_s, dc_lines = plot_side(dc_entry, 'DC')
    active_lines = ac_lines if d.ACorDC == 'AC' else dc_lines

    available_depth_arrays = [df['True_depth(m)'].to_numpy()]
    if ac_depth_s is not None:
        available_depth_arrays.append(ac_depth_s)
    if dc_depth_s is not None:
        available_depth_arrays.append(dc_depth_s)
    depth_concat = np.concatenate(available_depth_arrays)
    ylim_max = np.nanmax(depth_concat) + 0.1
    ylim_min = np.nanmin(depth_concat) - 0.1
    for axis in all_data_axes:
        axis.set_ylim(ylim_max, ylim_min)

    active_side_axes = pair_axes[d.ACorDC]
    for axis in all_data_axes:
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)
            spine.set_edgecolor('black')
    for axis in active_side_axes:
        for spine in axis.spines.values():
            spine.set_linewidth(2.5)
            spine.set_edgecolor('red')

    mark_colors = {
        'surface_imperfection': 'gray',
        'tephra_or_layer': 'orange',
        'rock_or_debris': 'purple'
    }
    marker_keys = set()

    # Data structures to track selections and patches
    selections = {yv: [] for yv in y_values}
    visible_patches = {yv: [] for yv in y_values}
    current_idx = 0
    current_y = y_values[current_idx]

    def get_active_mark():
        if 'a' in marker_keys:
            return 'rock_or_debris'
        if 't' in marker_keys:
            return 'tephra_or_layer'
        return 'surface_imperfection'

    def update_title():
        active_mark = get_active_mark()
        # Simple header: Running AC or Running DC
        header_main = f"Running {d.ACorDC}"
        if d.ACorDC == 'AC':
            # AC now running, DC reference
            side_desc = "AC (now running), DC (for reference)"
        else:
            side_desc = "AC (for reference), DC (now running)"
        fig.suptitle(
            (
                f"{header_main} — {d.core}, section {d.section}; {side_desc}; "
                f"track {current_idx + 1} of {len(y_values)} (Y={current_y}); "
                f"drag mode: {active_mark}; file {entry_number} of {total_entries}"
            ),
            y=0.98
        )

    def recompute_mark_columns(y_value):
        y_mask = df['Y_dimension(mm)'] == y_value
        df.loc[y_mask, ['surface_imperfection', 'tephra_or_layer', 'rock_or_debris']] = 0
        for selection in selections[y_value]:
            m = (
                y_mask &
                (df['True_depth(m)'] >= selection['vmin']) &
                (df['True_depth(m)'] <= selection['vmax'])
            )
            df.loc[m, selection['mark']] = 1

    def redraw_current_y_patches():
        for patch_set in visible_patches[current_y]:
            for patch in patch_set:
                patch.remove()
        visible_patches[current_y].clear()
        for selection in selections[current_y]:
            color = mark_colors[selection['mark']]
            left_patch = ax_ac_left.axhspan(selection['vmin'], selection['vmax'], facecolor=color, alpha=0.3)
            right_patch = ax_ac_right.axhspan(selection['vmin'], selection['vmax'], facecolor=color, alpha=0.3)
            left_patch_ref = ax_dc_left.axhspan(selection['vmin'], selection['vmax'], facecolor=color, alpha=0.3)
            right_patch_ref = ax_dc_right.axhspan(selection['vmin'], selection['vmax'], facecolor=color, alpha=0.3)
            visible_patches[current_y].append((left_patch, right_patch, left_patch_ref, right_patch_ref))

    # Highlight the initial current Y line and title
    if current_y in active_lines:
        active_lines[current_y].set_linewidth(4)
        active_lines[current_y].set_zorder(2)
    update_title()

    # Callback for span selection
    def on_select(vmin, vmax):

        nonlocal current_y

        if vmax < vmin:
            vmin, vmax = vmax, vmin

        selections[current_y].append(
            {
                'vmin': vmin,
                'vmax': vmax,
                'mark': get_active_mark()
            }
        )
        recompute_mark_columns(current_y)
        redraw_current_y_patches()
        fig.canvas.draw_idle()

    # Callback for key presses
    def on_key_press(event):
        nonlocal current_idx, current_y
        key = event.key
        if key in {'t', 'a'}:
            marker_keys.add(key)
            update_title()
            fig.canvas.draw_idle()
            return

        if key == 'd':
            # Delete last selection for current Y
            if selections[current_y]:
                selections[current_y].pop()
                recompute_mark_columns(current_y)
                redraw_current_y_patches()
                fig.canvas.draw_idle()

        elif key == 'n':
            # Advance to next Y-dimension
            if current_idx < len(y_values) - 1:
                for patch_set in visible_patches[current_y]:
                    for patch in patch_set:
                        patch.remove()
                visible_patches[current_y].clear()
                current_idx += 1
                current_y = y_values[current_idx]
                # Reset all lines and highlight the new current Y line
                for yv, line in active_lines.items():
                    line.set_linewidth(0.5)
                    line.set_zorder(1)
                if current_y in active_lines:
                    active_lines[current_y].set_linewidth(4)
                    active_lines[current_y].set_zorder(2)
                redraw_current_y_patches()
                update_title()
                fig.canvas.draw_idle()
            else:
                print("    Last Y reached. Press 'w' to save or 'r' to skip.")

        elif key == 'b':
            # Advance to previous Y-dimension
            if current_idx > 0:
                for patch_set in visible_patches[current_y]:
                    for patch in patch_set:
                        patch.remove()
                visible_patches[current_y].clear()
                current_idx -= 1
                current_y = y_values[current_idx]
                # Reset all lines and highlight the new current Y line
                for yv, line in active_lines.items():
                    line.set_linewidth(0.5)
                    line.set_zorder(1)
                if current_y in active_lines:
                    active_lines[current_y].set_linewidth(4)
                    active_lines[current_y].set_zorder(2)
                redraw_current_y_patches()
                update_title()
                fig.canvas.draw_idle()
            else:
                print("    Can't go back from first Y.")

        elif key == 'w':
            # Save and move on
            output_dir = os.path.join(path_to_output_data, d.core)
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(os.path.join(output_dir, fname), index=False)
            print(f"    Saved {fname}")
            plt.close(fig)

        elif key == 'r':
            # Skip without saving
            print(f"    Skipped {fname}")
            plt.close(fig)
        elif key == 'q':
            # Quit the program
            print("    Quitting...")
            plt.close(fig)
            sys.exit()

    def on_key_release(event):
        key = event.key
        if key in {'t', 'a'}:
            marker_keys.discard(key)
            update_title()
            fig.canvas.draw_idle()

    # Attach span selectors and keypress handler
    span_ac_left = SpanSelector(ax_ac_left, on_select, 'vertical',
                                useblit=True, props=dict(facecolor='gray', alpha=0.3),
                                interactive=False)
    span_ac_right = SpanSelector(ax_ac_right, on_select, 'vertical',
                                 useblit=True, props=dict(facecolor='gray', alpha=0.3),
                                 interactive=False)
    span_dc_left = SpanSelector(ax_dc_left, on_select, 'vertical',
                                useblit=True, props=dict(facecolor='gray', alpha=0.3),
                                interactive=False)
    span_dc_right = SpanSelector(ax_dc_right, on_select, 'vertical',
                                 useblit=True, props=dict(facecolor='gray', alpha=0.3),
                                 interactive=False)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)

    plt.show()

#%% Run on all ECM datasets
for idx, entry in enumerate(data, start=1):
    process_ecm(entry, idx, len(data))
