"""Jane's molecular graph, the trained message rounds, and a final prediction."""

import math
import numpy as np
import pygame
from rdkit import Chem
from rdkit.Chem import AllChem

width, height = 1480, 940
background = (255, 255, 255)
text_color = (255, 255, 255)
black = (0, 0, 0)
line_color = (90, 90, 90)
selection_color = (65, 65, 65)
# Three background source groups and three individually tracked atoms.
source_colors = ((141, 83, 204), (32, 115, 224), (232, 131, 15),
                 (0, 148, 103), (218, 63, 113), (0, 150, 190))
element_colors = {
    "C": (85, 85, 85), "N": (48, 80, 248),
    "O": (224, 32, 32), "S": (224, 192, 32),
}
other_color = (187, 136, 221)


def blend(a, b, fraction):
    fraction = max(0, min(1, float(fraction)))
    return tuple(int(x + (y - x) * fraction) for x, y in zip(a, b))


def font(app, size, bold=False):
    key = (size, bold)
    if key not in app.fonts:
        app.fonts[key] = pygame.font.SysFont(None, size, bold=bold)
    return app.fonts[key]


def prepare_molecule(app):
    # Jane's RDKit coordinate loop, fitted to the larger graph area.
    app.mol = Chem.Mol(app.trace["mol"])
    AllChem.Compute2DCoords(app.mol)
    conf = app.mol.GetConformer()
    app.n = app.mol.GetNumAtoms()
    coords = []
    for i in range(app.n):
        coords.append([conf.GetAtomPosition(i).x, -conf.GetAtomPosition(i).y])
    coords = np.array(coords)
    extent = np.maximum(np.ptp(coords, axis=0), 0.2)
    space = np.array([app.plot_rect.width - 120, app.plot_rect.height - 120])
    scale = min(100, min(space / extent))
    midpoint = (coords.max(0) + coords.min(0)) / 2
    app.positions = (coords - midpoint) * scale + app.plot_rect.center
    app.nodes = {i: (int(x), int(y)) for i, (x, y) in enumerate(app.positions)}
    app.bonds = []
    for edge in app.mol.GetBonds():
        app.bonds.append((edge.GetBeginAtomIdx(), edge.GetEndAtomIdx(),
                          max(1, int(edge.GetBondTypeAsDouble()))))
    app.radius = 25 if app.n <= 36 else 21 if app.n <= 65 else 16
    lengths = [np.linalg.norm(app.positions[i] - app.positions[j]) for i, j, _ in app.bonds]
    if lengths:
        app.radius = min(app.radius, max(5, int(min(lengths) * 0.23)))


def prepare_flow(app):
    """Track three atoms individually and group the other sources by color."""
    adjacency = app.trace["adjacency"]
    focus = app.selected_atom
    bridge = min(app.one_hop) if app.one_hop else None
    upstream = None
    for neighbor in sorted(app.one_hop):
        candidates = sorted(set(np.flatnonzero(adjacency[neighbor]).tolist()) & app.two_hop)
        if candidates:
            bridge, upstream = neighbor, candidates[0]
            break
    app.flow_path = (upstream, bridge, focus)
    initial = [frozenset({(0, 4, 5)[index % 3]}) for index in range(app.n)]
    for code, index in ((1, focus), (2, bridge), (3, upstream)):
        if index is not None:
            initial[index] = frozenset({code})
    app.flow_origins = [initial]
    for _ in range(app.layers):
        previous = app.flow_origins[-1]
        updated = []
        for index in range(app.n):
            origins = set(previous[index])
            for neighbor in np.flatnonzero(adjacency[index]):
                origins.update(previous[neighbor])
            updated.append(frozenset(origins))
        app.flow_origins.append(updated)


def draw_text(surface, text, font, color, center):
    # Jane's centered text helper, used for her white atom indices.
    image = font.render(text, True, color)
    surface.blit(image, image.get_rect(center=center))


def draw_bond(surface, start, end, order):
    # Jane's perpendicular offsets for single, double, and triple bonds.
    p, q = pygame.Vector2(start), pygame.Vector2(end)
    direction = (q - p).normalize()
    normal = pygame.Vector2(-direction.y, direction.x)
    offsets = {1: [0], 2: [-4, 4], 3: [-7, 0, 7]}[order]
    for offset in offsets:
        pygame.draw.line(surface, line_color, p + normal * offset, q + normal * offset, 2)


def round_progress(app):
    duration = app.round_duration
    arrival = duration * 0.69
    update = np.clip((app.animation_time - arrival) / (duration - arrival), 0, 1)
    return float(update)


def message_paths(app, moment):
    """Packets use one-hop edges and the input vector for the current layer."""
    start_time, arrival = app.round_duration * 0.12, app.round_duration * 0.69
    if not 1 <= app.stage <= app.layers or not start_time <= moment < arrival:
        return []
    fraction = (moment - start_time) / (arrival - start_time)
    values = app.trace["features"] if app.stage == 1 else app.trace["hidden_states"][app.stage - 2]
    packets = []
    for i, j, _ in app.bonds:
        for source, target in ((i, j), (j, i)):
            delta = app.positions[target] - app.positions[source]
            length = np.linalg.norm(delta)
            direction = delta / length
            normal = np.array([-direction[1], direction[0]])
            padding = min(app.radius + 14, length * 0.35)
            start = app.positions[source] + direction * padding
            end = app.positions[target] - direction * padding
            position = start * (1 - fraction) + end * fraction + normal * 10
            packets.append((source, target, position, values[source]))
    return packets


def highlighted(app, source, target):
    upstream, bridge, focus = app.flow_path
    if app.stage == 1:
        return (source, target) == (upstream, bridge)
    return app.stage == 2 and (source, target) == (bridge, focus)


def draw_sectors(surface, position, radius, origins, intensity=1, width=0):
    """Equal sectors show category presence, not weights or feature values."""
    colors = sorted(origins)
    angle_size = 2 * math.pi / len(colors)
    rect = pygame.Rect(position[0] - radius, position[1] - radius, 2 * radius, 2 * radius)
    for part, code in enumerate(colors):
        color = blend(background, source_colors[code], intensity)
        start, end = part * angle_size, (part + 1) * angle_size
        if width:
            pygame.draw.arc(surface, color, rect, start, end, width)
        else:
            points = [position]
            for angle in np.linspace(start, end, 16):
                points.append((position[0] + radius * math.cos(angle),
                               position[1] - radius * math.sin(angle)))
            pygame.draw.polygon(surface, color, points)
    if not width:
        pygame.draw.circle(surface, blend(background, line_color, intensity), position, radius, 1)


def draw_messages(app):
    previous = app.flow_origins[app.stage - 1]
    for source, target, position, _ in message_paths(app, app.animation_time):
        on_path = highlighted(app, source, target)
        to_focus = target == app.selected_atom
        radius = 12 if on_path else 9 if to_focus else 5
        intensity = 1 if on_path or to_focus else 0.7
        # Every round copies the source's complete previous-round color set.
        draw_sectors(app.surface, position, radius, previous[source], intensity)


def draw_origin_ring(app, index, position, update):
    arrival = app.round_duration * 0.69
    layer = min(app.stage, app.layers)
    if 1 <= app.stage <= app.layers and app.animation_time < arrival:
        layer -= 1
    origins = app.flow_origins[layer][index]
    pulse = int(3 * math.sin(math.pi * update)) if 1 <= app.stage <= app.layers else 0
    radius = app.radius + max(3, min(6, app.radius // 4)) + pulse
    width = max(2, min(4, app.radius // 4))
    draw_sectors(app.surface, position, radius, origins, width=width)


def stage_title(app):
    if app.stage == 0:
        return "Molecule as the model sees it"
    if app.stage <= app.layers:
        return "Message round " + str(app.stage)
    return "Prediction"


def draw_header(app):
    # once a round's packets have landed, rounds 1 and 2 leave almost the same picture,
    # so without this line there is nothing on screen saying which step you are on
    draw_text(app.surface, "Step " + str(app.stage) + " of " + str(app.prediction_stage)
              + "    " + stage_title(app), font(app, 36, bold=True), (40, 40, 40),
              (width / 2, 44))
    pool = app.pool().tolist()
    place = pool.index(app.index) + 1 if app.index in pool else 1
    line = (str(app.bundle["names"][app.index]) + "    test molecule " + str(place)
            + " of " + str(len(pool)) + "    atom " + str(app.selected_atom) + " in focus")
    draw_text(app.surface, line, font(app, 26), (120, 120, 120), (width / 2, 80))
    draw_text(app.surface, "Space: next step    R: reset    Arrows: molecule    Click: an atom",
              font(app, 22), (170, 170, 170), (width / 2, 922))


def draw_molecule(app):
    draw_header(app)
    update = round_progress(app) if 1 <= app.stage <= app.layers else 0
    # Jane draws bonds first, then atom circles and their centered labels.
    for i, j, order in app.bonds:
        draw_bond(app.surface, app.nodes[i], app.nodes[j], order)
    if 1 <= app.stage <= app.layers:
        draw_messages(app)
    for i, position in app.nodes.items():
        atom = app.mol.GetAtomWithIdx(i)
        tint = element_colors.get(atom.GetSymbol(), other_color)
        pygame.draw.circle(app.surface, tint, position, app.radius)
        pygame.draw.circle(app.surface, black, position, app.radius, 2)
        draw_origin_ring(app, i, position, update)
        if i == app.selected_atom:
            pygame.draw.circle(app.surface, selection_color, position, app.radius + 17, 1)
        size = max(10, int(app.radius * 0.85))
        draw_text(app.surface, str(i), font(app, size), text_color, position)


def predicted_label(score, threshold):
    return "active" if score >= threshold else "inactive"


def draw_prediction(app):
    # Fade in the real result; never count through invented intermediate scores.
    opacity = float(np.clip(app.animation_time / app.round_duration, 0, 1))
    if opacity == 0:
        return
    score = app.trace["probability"]
    label = predicted_label(score, app.threshold)
    # the measured label only appears here, at the end, so the answer is not sitting
    # on screen while the model is still passing messages
    measured = "active" if app.labels[app.index] == 1 else "inactive"
    result = (f"Model score {score:.3f}   |   Predicted " + label
              + f"   |   Threshold {app.threshold:.3f}   |   Measured " + measured)
    color = blend(background, (40, 40, 40), opacity)
    draw_text(app.surface, result, font(app, 28), color, (width / 2, 886))
