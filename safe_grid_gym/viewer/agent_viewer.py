"""
The AgentViewer class provides a way to display gridworlds in the terminal.

AgentViewer was created by n0p2.
The original repo can be found at https://github.com/n0p2/ai-safety-gridworlds-viewer

It is used in the gym environment to render the environment for human.
The code is integrated into this repo to simplify dependency management.
Note, that this rendering does not fully work in python 2.
"""

import logging
import collections
import six
import curses
import datetime
import time


class AgentViewer(object):
    """A terminal-based game viewer for ai-safety-gridworlds games.
  (https://github.com/deepmind/ai-safety-gridworlds)

  This is based on the `human_ui.CursesUi` class from the pycolab game
  engine (https://github.com/deepmind/pycolab) developped by Deepmind.
  Both `CursesUi` and its subclass `safety_ui.SafetyCursesUi` allow a
  human player to play their games with keyboard input.

  `AgentViewer` is created to enable display of a live game as an agent
  plays it. This is desirable in reinforcement learning (RL) settings,
  where one need to view an agent's interactions with the environment
  as the game progresses.

  As far as programming paradigm goes, I try to find a balance
  between OO (object oriented) and FP (functional programming).
  Classes are defined to manage resources and mutable states.
  All other resuable logic are defined as functions (without side
  effect) outside of the class.
  """

    def __init__(self, pause, **kwargs):
        """Construct an `AgentViewer`, which displays agent's interactions
    with the environment in a terminal for ai-safety-gridworlds games
    developed by Google Deepmind.

    Args:
      pause: float.
          A game played by an agent often proceed at a pace too fast
          for meaningful watching. `pause` allows one to adjust the
          displaying pace. Note that when displaying an elapsed time
          on the game window, the wall clock time consumed by pausing
          is subtracted (see `_get_elapsed`).

    """
        self._screen = curses.initscr()
        self._colour_pair = init_curses(self._screen, **kwargs)
        self._pause = pause
        self.reset_time()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def close(self):
        curses.endwin()

    def display(self, env):
        """
    Args:
      env: ai_safety_gridworlds.environments.shared.safety_game.SafetyEnvironment.
          An instance of SafetyEnvironment which contains observations (or boards)
          and returns.
    """

        board = env.current_game._board.board
        return_ = env.episode_return
        # Time cost is not always a good indicator for performance evaluation.
        # Other indicators, such as number of episodes, might be more suitable.
        # Neverthelesss, only elapsed time is displayed, while support of
        # additional information should be done by the consumer of AgentViewer.
        elapsed = self._get_elapsed()
        try:
            display(self._screen, board, return_, elapsed, self._colour_pair)
            self._do_pause()
            if self._pause is not None:
                time.sleep(self._pause)
        except:
            curses.endwin()

    def reset_time(self):
        self._start_time = time.time()
        self._pause_cnt = 0

    def _do_pause(self):
        if self._pause is not None:
            time.sleep(self._pause)
            self._pause_cnt += 1

    def _get_elapsed(self):
        now = time.time()
        s = 0.0 if self._pause is None else self._pause
        elapsed = now - self._start_time - float(s) * self._pause_cnt
        return elapsed


# --------
# Core functions that deal with screen initialization and display.
# These functions are heavily based on the `human_ui.CursesUi` class
# (https://github.com/deepmind/pycolab/blob/master/pycolab/human_ui.py)
# --------


def display(screen, board, score, elapsed, color_pair):
    """Redraw the game board onto the already-running screen, with elapsed time and score.

  Args:
    screen
    obs: TODO class. A `???` object containing the current game board.
    score:
    elapsed: seconds
  """
    # screen.erase()
    screen.clear()

    # Display the game clock and the current score.
    screen.addstr(0, 2, ts2str(elapsed), curses.color_pair(0))
    screen.addstr(0, 10, "Score: %.2f" % score, curses.color_pair(0))

    # Display game board rows one-by-one.
    for row, board_line in enumerate(board, start=1):
        screen.move(row, 0)  # Move to start of this board row.
        # Display game board characters one-by-one.
        for character in board_line:
            character = int(character)
            color_id = color_pair[chr(character)]
            color_ch = curses.color_pair(color_id)
            screen.addch(character, color_ch)

    screen.refresh()


def init_colour(color_bg, color_fg):
    """
  Based on `human_ui.CursesUi._init_colour`
  (https://github.com/deepmind/pycolab/blob/master/pycolab/human_ui.py)
  """
    curses.start_color()
    # The default colour for all characters without colours listed is boring
    # white on black, or "system default", or somesuch.
    colour_pair = collections.defaultdict(lambda: 0)
    # And if the terminal doesn't support true color, that's all you get.
    if not curses.can_change_color():
        return colour_pair

    # Collect all unique foreground and background colours. If this terminal
    # doesn't have enough colours for all of the colours the user has supplied,
    # plus the two default colours, plus the largest colour id (which we seem
    # not to be able to assign, at least not with xterm-256color) stick with
    # boring old white on black.
    colours = set(six.itervalues(color_fg)).union(six.itervalues(color_bg))
    if (curses.COLORS - 2) < len(colours):
        return colour_pair

    # Get all unique characters that have a foreground and/or background colour.
    # If this terminal doesn't have enough colour pairs for all characters plus
    # the default colour pair, stick with boring old white on black.
    characters = set(color_fg).union(color_bg)
    if (curses.COLOR_PAIRS - 1) < len(characters):
        return colour_pair

    # Get the identifiers for both colours in the default colour pair.
    cpair_0_fg_id, cpair_0_bg_id = curses.pair_content(0)

    # With all this, make a mapping from colours to the IDs we'll use for them.
    ids = set(range(curses.COLORS - 1)) - {  # The largest ID is not assignable?
        cpair_0_fg_id,
        cpair_0_bg_id,
    }  # We don't want to change these.
    ids = list(reversed(sorted(ids)))  # We use colour IDs from large to small.
    ids = ids[: len(colours)]  # But only those colour IDs we actually need.
    colour_ids = dict(zip(colours, ids))

    # Program these colours into curses.
    for colour, cid in six.iteritems(colour_ids):
        curses.init_color(cid, *colour)

    # Now add the default colours to the colour-to-ID map.
    cpair_0_fg = curses.color_content(cpair_0_fg_id)
    cpair_0_bg = curses.color_content(cpair_0_bg_id)
    colour_ids[cpair_0_fg] = cpair_0_fg_id
    colour_ids[cpair_0_bg] = cpair_0_bg_id

    # The color pair IDs we'll use for all characters count up from 1; note that
    # the "default" colour pair of 0 is already defined, since _colour_pair is a
    # defaultdict.
    colour_pair.update(
        {character: pid for pid, character in enumerate(characters, start=1)}
    )

    # Program these color pairs into curses, and that's all there is to do.
    for character, pid in six.iteritems(colour_pair):
        # Get foreground and background colours for this character. Note how in
        # the absence of a specified background colour, the same colour as the
        # foreground is used.
        cpair_fg = color_fg.get(character, cpair_0_fg_id)
        cpair_bg = color_bg.get(character, cpair_fg)
        # Get colour IDs for those colours and initialise a colour pair.
        cpair_fg_id = colour_ids[cpair_fg]
        cpair_bg_id = colour_ids[cpair_bg]
        curses.init_pair(pid, cpair_fg_id, cpair_bg_id)

    return colour_pair


def char2ord_4_colormap(colour):
    if colour is not None:
        return {ord(char): colour for char, colour in six.iteritems(colour)}
    else:
        return None


def init_curses(screen, color_bg, color_fg, delay=None):
    logger = get_logger()
    logger.info("init_curses...")
    # If the terminal supports colour, program the colours into curses as
    # "colour pairs". Update our dict mapping characters to colour pairs.
    colour_pair = init_colour(color_bg, color_fg)
    curses.curs_set(0)  # We don't need to see the cursor.
    if delay is None:
        screen.timeout(-1)  # Blocking reads
    else:
        screen.timeout(delay)  # Nonblocking (if 0) or timing-out reads

    logger.info("init_curses success.")
    return colour_pair


def ts2str(ts_delta):
    delta = datetime.timedelta(seconds=ts_delta)
    return str(delta).split(".")[0]


# --------
# logging and debugging
# --------
_logger = None


def get_logger():
    """
  singleton
  """
    global _logger
    if _logger is None:
        _logger = logging.getLogger(__file__)
        hdlr = logging.FileHandler(__file__ + ".log")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        hdlr.setFormatter(formatter)
        _logger.addHandler(hdlr)
        _logger.setLevel(logging.DEBUG)

    return _logger
