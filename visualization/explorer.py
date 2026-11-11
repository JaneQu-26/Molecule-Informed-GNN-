"""Present each trained message-passing layer, then the prediction."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np
import pygame
from rdkit import Chem

from gnn.predict import load_bundle, trace_molecule, validate_bundle
from . import drawing


class Explorer:
    # one Explorer is one molecule on screen: its trace, which atom is in focus,
    # which step we are on, and how far into that step the animation has played
    def __init__(self, bundle, index=None, stage=0, atom=None):
        self.bundle = bundle
        self.smiles = bundle["smiles"]
        self.labels = bundle["labels"]
        self.splits = bundle["splits"]
        # the score PyTorch gave each molecule in the notebook, used for the opening pick
        self.reference = bundle["reference_probabilities"]
        # the 0.688 the notebook chose on validation, not 0.5
        self.threshold = float(bundle["decision_threshold"])
        # how many seconds one message-passing round takes on screen
        self.round_duration = 3.2
        self.fonts = {}
        # we draw on a fixed-size surface and scale it to the window at the end,
        # so every coordinate in drawing.py can be a plain number
        self.surface = pygame.Surface((drawing.width, drawing.height))
        self.plot_rect = pygame.Rect(90, 90, 1300, 760)
        index = self.opening_example() if index is None else index
        if not 0 <= index < len(self.smiles):
            raise ValueError("Molecule index is outside this model's dataset")
        self.set_molecule(index)
        if atom is not None:
            self.select_atom(atom)
        self.change_stage(stage)

    def pool(self):
        # we browse test molecules only, because those are the ones the model never saw.
        # the fallback is for a bundle with no test split, which only happens in a check
        test = np.flatnonzero(self.splits == "test")
        return test if len(test) else np.arange(len(self.smiles))

    def opening_example(self):
        # Start with a compact, correctly predicted active test molecule.
        # it is still a real test molecule, we only pick one small enough to read
        candidates = [i for i in self.pool()
                      if self.labels[i] == 1 and self.reference[i] >= self.threshold]
        candidates.sort(key=lambda i: self.reference[i], reverse=True)
        for index in candidates[:150]:
            mol = Chem.MolFromSmiles(str(self.smiles[index]))
            if 15 <= mol.GetNumAtoms() <= 38:
                return int(index)
        return int(self.pool()[0])

    def set_molecule(self, index):
        self.index = int(index)
        # here is where the model actually runs. everything drawn later reads this dict
        self.trace = trace_molecule(str(self.smiles[index]), self.bundle)
        self.layers = len(self.trace["hidden_states"])
        # two layers, so the stages are 0 static, 1 and 2 the rounds, 3 the prediction
        self.prediction_stage = self.layers + 1
        drawing.prepare_molecule(self)
        # we open on the busiest atom, because its ring is the one worth watching
        degrees = self.trace["adjacency"].sum(1)
        self.select_atom(int(np.argmax(degrees)))

    def select_atom(self, atom):
        self.selected_atom = int(np.clip(atom, 0, self.n - 1))
        adjacency = self.trace["adjacency"]
        # one hop is who this atom hears from in round 1, two hops is who it reaches
        # by round 2. this is the receptive field the rings are drawing
        self.one_hop = set(np.flatnonzero(adjacency[self.selected_atom]).tolist())
        two_hops = set()
        for neighbor in self.one_hop:
            two_hops.update(np.flatnonzero(adjacency[neighbor]).tolist())
        self.two_hop = two_hops - self.one_hop - {self.selected_atom}
        drawing.prepare_flow(self)
        # changing the focus restarts the walkthrough, otherwise the rings on screen
        # would belong to the old atom
        self.change_stage(0)

    def change_stage(self, stage):
        if not 0 <= stage <= self.prediction_stage:
            raise ValueError("Stage must be between zero and " + str(self.prediction_stage))
        self.stage = stage
        self.animation_time = 0.0
        self.playing = stage > 0

    def advance(self):
        # Finish the current step before accepting another space press.
        # without this, holding space would skip a whole round
        if not self.playing:
            self.change_stage((self.stage + 1) % (self.layers + 2))

    def update(self, elapsed):
        if self.playing:
            self.animation_time = min(self.round_duration, self.animation_time + elapsed)
            self.playing = self.animation_time < self.round_duration

    def next_molecule(self, direction=1):
        # the arrow keys walk through the test set and wrap around at either end
        pool = self.pool().tolist()
        position = pool.index(self.index) if self.index in pool else 0
        self.set_molecule(pool[(position + direction) % len(pool)])

    def viewport(self, window_size):
        # the fixed surface is letterboxed into whatever size the window is now
        scale = min(window_size[0] / drawing.width, window_size[1] / drawing.height)
        rect = pygame.Rect(0, 0, int(drawing.width * scale), int(drawing.height * scale))
        rect.center = (window_size[0] // 2, window_size[1] // 2)
        return rect

    def handle_event(self, event, window_size=(drawing.width, drawing.height)):
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: return False
            if event.key == pygame.K_SPACE: self.advance()
            elif event.key == pygame.K_r: self.change_stage(0)
            elif event.key == pygame.K_LEFT: self.next_molecule(-1)
            elif event.key == pygame.K_RIGHT: self.next_molecule()
        # clicking an atom moves the focus, but not on the prediction screen,
        # where there is no ring left to redraw
        if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.stage < self.prediction_stage):
            rect = self.viewport(window_size)
            if rect.collidepoint(event.pos):
                # the click lands in window coordinates, so we map it back onto the
                # fixed surface before measuring distances to the atoms
                point = ((event.pos[0] - rect.x) * drawing.width / rect.width,
                         (event.pos[1] - rect.y) * drawing.height / rect.height)
                distances = np.linalg.norm(self.positions - point, axis=1)
                # a little slack past the circle, so near misses still count
                if distances.min() <= self.radius + 12:
                    self.select_atom(int(distances.argmin()))
        return True

    def render(self, animation_time=None):
        # passing a time renders that exact moment, which is how screenshots are taken
        if animation_time is not None:
            self.animation_time = max(0, min(animation_time, self.round_duration))
            self.playing = self.stage > 0 and self.animation_time < self.round_duration
        self.surface.fill(drawing.background)
        drawing.draw_molecule(self)
        if self.stage == self.prediction_stage:
            drawing.draw_prediction(self)
        return self.surface


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=root / "models/model.npz")
    parser.add_argument("--stage", type=int, default=0, help="0: initial; 1..N: layers; N+1: prediction")
    parser.add_argument("--index", type=int, help="Molecule index, starting at zero")
    parser.add_argument("--atom", type=int, help="Atom index, starting at zero")
    parser.add_argument("--time", type=float, default=1.25, help="Animation time in a screenshot")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)
    # the dummy drivers let a screenshot be taken with no window and no sound device
    if args.screenshot:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    bundle = load_bundle(args.model)
    # we check against PyTorch before drawing anything, so a mismatch is never shown
    print("Viewer/model maximum score difference:", validate_bundle(bundle))
    pygame.init()
    try:
        if args.screenshot:
            pygame.display.set_mode((1, 1))
        else:
            # we shrink to fit the screen, because the surface is 1480 by 940
            desktop = pygame.display.get_desktop_sizes()[0]
            scale = min(1, desktop[0] * 0.94 / drawing.width, desktop[1] * 0.88 / drawing.height)
            window = pygame.display.set_mode((int(drawing.width * scale), int(drawing.height * scale)), pygame.RESIZABLE)
            pygame.display.set_caption("Molecule GNN")
            print("Space: next step | R: reset | Arrows: molecule | Click: focus atom | Esc: quit")
        app = Explorer(bundle, args.index, args.stage, args.atom)
        print("Model:", app.layers, "message-passing layers; prediction at step", app.prediction_stage)
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(app.render(args.time), str(args.screenshot))
            print("Screenshot:", args.screenshot.resolve())
            return
        # Jane's event -> draw -> flip -> clock loop.
        clock = pygame.time.Clock()
        running, frames = True, 0
        while running:
            # the cap keeps one slow frame from jumping the animation forward
            elapsed = min(clock.tick(45) / 1000, 0.1)
            for event in pygame.event.get():
                if event.type == pygame.VIDEORESIZE:
                    window = pygame.display.set_mode((max(800, event.w), max(510, event.h)), pygame.RESIZABLE)
                running = app.handle_event(event, window.get_size()) and running
            app.update(elapsed)
            viewport = app.viewport(window.get_size())
            window.fill(drawing.background)
            window.blit(pygame.transform.smoothscale(app.render(), viewport.size), viewport)
            pygame.display.flip()
            frames += 1
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
