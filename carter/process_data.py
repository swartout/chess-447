# Script for processing data.
#     Processes raw pgn data and pickles data.
#     Pickled format is `data, char_emb, board_emb = pickle.load(file)`
#     get help with `python process_data.py -h`

import chess.pgn
import io
from typing import Dict, List, Tuple
import torch
import argparse
import pickle


BOARD_EMB = {
    0: "P",
    1: "R",
    2: "N",
    3: "B",
    4: "Q",
    5: "K",
    6: "p",
    7: "r",
    8: "n",
    9: "b",
    10: "q",
    11: "k",
}
BAD_CHARS = set(["%", "l", "!", "?", "v", "}", "]", "{", "["])


def load_games(filename: str, verbose=False) -> Tuple[List[str], List[chess.pgn.Game]]:
    """Load games and pgn from a file into chess.pgn.Games.

    Args:
        - filename: where the pgn file is (lichess format)
        - verbose: verbose logging option

    Returns:
        - tuple of (list of str pgns, list of chess.pgn.Game objects)
    """
    if verbose:
        print(f"Loading from: {filename}")
    bad_chars = set(["%", "l", "!", "?", "v", "}", "]", "{", "["])
    pgns = []
    with open(filename, "r") as f:
        for row in f:
            if row[:2] == "1." and all(char not in row for char in bad_chars):
                pgns.append(row)

    games = []
    percent = len(pgns) // 100
    with io.StringIO("\n".join(pgns)) as f:
        for i in range(len(pgns)):
            if verbose and i % percent == 0:
                print(f"{(100 * i / len(pgns)):.0f}% done...")
            games.append(chess.pgn.read_game(f))
    return pgns, games


def get_char_emb(pgns: List[str]) -> Dict[str, int]:
    """Get a character embedding dictionary from a list of pgns.

    Args:
        - games: list of pgn strings

    Returns:
        - dictionary mapping each char in the pgns to an int
    """
    chars = set()
    for i, game in enumerate(pgns):
        if all(char not in game for char in BAD_CHARS):
            for char in game:
                chars.add(char)
    return {c: i for i, c in enumerate(chars)}


def board_to_list(board: chess.Board, board_emb: Dict[int, str]) -> List[int]:
    """Get a 12*8*8 one-hot representation of a board.

    Args:
        - board: chess.Board to be converted
        - board_emb: dict mapping piece representations to board level indicies

    Returns:
        - 12*8*8 one-hot board representation
    """
    out = []
    for i in range(12):
        board_rep = str(board).replace("\n", "").replace(" ", "")
        out += [1 if c == board_emb[i] else 0 for c in board_rep]
    return out


def process_game(
    game: chess.pgn.Game, char_emb: Dict[str, int], board_emb: Dict[int, str]
) -> Dict[str, torch.Tensor]:
    """Returns a dict of PGN string, board state, and move indicies.

    Args:
        - game: chess.pgn.Game to parse
        - char_emb: embedding dictionary converting chars to ints
        - board_emb: embedding from piece char to board index

    Returns:
        - a dictionary of results, str to tensor:
            pgn: tensor of indicies of the chars in the pgn string
            boards: tensor of board states, (N_moves X 12*8*8)
            move_idx: tensor mapping moves to pgn indices (N_moves,)
    """
    pgn_str = str(game).split("\n\n")[1]
    pgn = torch.Tensor([char_emb[c] for c in pgn_str])
    moves = pgn_str.split(" ")

    move_idx = []
    str_len = 0
    for i, m in enumerate(moves[:-1]):
        str_len += len(m)
        if i % 3 != 0:
            move_idx.append(str_len)
        str_len += 1

    chess_board = game.board()
    board_states = []
    for move in game.mainline_moves():
        chess_board.push(move)
        board_states.append(board_to_list(chess_board, board_emb))

    assert len(board_states) == len(
        move_idx
    ), f"board_len: {len(board_states)}, move_idx_len: {len(move_idx)}"
    return {
        "pgn": pgn,
        "boards": torch.Tensor(board_states),
        "move_idx": torch.Tensor(move_idx),
    }


def main(args):
    pgns, games = load_games(args.input, verbose=args.v)
    char_emb = get_char_emb(pgns)

    if args.v:
        print("Starting data processing:")
    data = []
    p = len(games) // 100
    for i in range(len(games)):
        data.append(process_game(games[i], char_emb, BOARD_EMB))
        if args.v and i % p == 0:
            print(f"{(100 * i / len(games)):.0f}% done...")

    if args.v:
        print(f"Saving processed data to: {args.output}")
    with open(args.output, "wb") as f:
        pickle.dump((data, char_emb, BOARD_EMB), f)
    if args.v:
        print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input path to raw data")
    parser.add_argument("output", help="output path for processed data")
    parser.add_argument("-v", help="verbose option", action="store_true")
    args = parser.parse_args()

    main(args)
