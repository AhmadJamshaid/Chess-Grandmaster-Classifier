"""
Chess GrandMaster Classifier - Streamlit App
Predicts which chess grandmaster played a game based on PGN input.
"""

import streamlit as st
import chess
import chess.pgn
import numpy as np
import pickle
import io
import os
import warnings
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chess GrandMaster Classifier",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  – dark chess aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary:   #0d0f14;
    --bg-card:      #13171f;
    --bg-elevated:  #1a1f2e;
    --accent-gold:  #f0c040;
    --accent-teal:  #00d4aa;
    --accent-blue:  #4f9eff;
    --accent-purple:#a78bfa;
    --text-primary: #f0f2f8;
    --text-muted:   #8892a4;
    --border:       rgba(240,192,64,0.18);
    --shadow:       0 8px 32px rgba(0,0,0,0.5);
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--accent-gold); border-radius: 3px; }

/* ── App container ── */
.main .block-container {
    padding: 2rem 3rem !important;
    max-width: 1400px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a1228 0%, #0d1829 40%, #12221a 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow);
}
.hero-banner::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% -20%, rgba(240,192,64,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f0c040, #ffd700, #fff8dc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
}
.hero-subtitle {
    color: var(--text-muted);
    font-size: 1.1rem;
    font-weight: 400;
    letter-spacing: 0.5px;
}
.chess-pieces-row {
    font-size: 2.2rem;
    letter-spacing: 0.5rem;
    margin-bottom: 1rem;
    opacity: 0.85;
}

/* ── Cards ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.6);
}
.card-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--accent-gold);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Player cards in sidebar ── */
.player-card {
    background: var(--bg-elevated);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.player-card:hover { border-color: var(--accent-gold); }
.player-name { font-weight: 600; font-size: 0.95rem; }
.player-era  { font-size: 0.75rem; color: var(--text-muted); margin-top: 2px; }

/* ── Result box ── */
.result-box {
    background: linear-gradient(135deg, #1a2a1a, #0d1f1a);
    border: 2px solid var(--accent-teal);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(0,212,170,0.15);
    animation: fadeInUp 0.6s ease;
}
.result-player {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4aa, #4f9eff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.5rem 0;
}
.result-confidence {
    font-size: 1.2rem;
    color: var(--accent-gold);
    font-weight: 600;
}
.result-color-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 1.2rem;
}

/* ── Probability bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0.7rem;
}
.prob-label {
    width: 100px;
    font-size: 0.88rem;
    font-weight: 500;
    text-align: right;
    color: var(--text-primary);
}
.prob-bar-bg {
    flex: 1;
    height: 10px;
    background: rgba(255,255,255,0.07);
    border-radius: 999px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 1s ease;
}
.prob-value {
    width: 52px;
    font-size: 0.85rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent-gold);
}

/* ── Feature badges ── */
.feat-badge {
    display: inline-block;
    background: var(--bg-elevated);
    border: 1px solid rgba(79,158,255,0.3);
    color: var(--accent-blue);
    border-radius: 6px;
    padding: 0.25rem 0.65rem;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 3px;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
    margin-left: 0.8rem;
}

/* ── Stat chips ── */
.stat-chip {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent-gold);
    font-family: 'JetBrains Mono', monospace;
}
.stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Streamlit overrides ── */
.stTextArea textarea {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent-gold) !important;
    box-shadow: 0 0 0 2px rgba(240,192,64,0.2) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #c8960c, #f0c040) !important;
    color: #0d0f14 !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: 0.5px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(240,192,64,0.35) !important;
}
.stSelectbox > div, .stFileUploader > div {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}
.stAlert { border-radius: 12px !important; }
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    padding: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-elevated) !important;
    color: var(--accent-gold) !important;
}
hr { border-color: var(--border) !important; }

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.6; }
}
.loading-pulse { animation: pulse 1.5s ease infinite; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION (identical to training)
# ─────────────────────────────────────────────────────────────────────────────

def extract_metadata_features(headers):
    features = {}
    event = headers.get('Event', '').lower()
    features['event_has_world']        = 1 if 'world' in event else 0
    features['event_has_championship'] = 1 if 'championship' in event or 'ch' in event else 0
    features['event_has_open']         = 1 if 'open' in event else 0
    features['event_has_tournament']   = 1 if 'tournament' in event or 'tourn' in event else 0
    features['event_has_match']        = 1 if 'match' in event else 0
    features['event_has_olympiad']     = 1 if 'olympiad' in event else 0
    features['event_length']           = min(len(event), 100)

    site = headers.get('Site', '').lower()
    features['site_has_usa']      = 1 if 'usa' in site or 'us ' in site or 'new york' in site or 'chicago' in site else 0
    features['site_has_europe']   = 1 if any(x in site for x in ['ger','fra','eng','ned','eur','london','paris','berlin']) else 0
    features['site_has_asia']     = 1 if any(x in site for x in ['chn','ind','jpn','asia','beijing','shanghai']) else 0
    features['site_has_russia']   = 1 if any(x in site for x in ['rus','soviet','ussr','moscow','leningrad']) else 0
    features['site_has_argentina']= 1 if 'arg' in site or 'argentina' in site or 'buenos aires' in site else 0
    features['site_length']       = min(len(site), 50)

    date_str = headers.get('Date', '????.??.??')
    try:
        if date_str != '????.??.??':
            parts = date_str.split('.')
            year  = int(parts[0]) if parts[0] != '????' else 0
            features['year']    = year if year > 0 else 1950
            features['month']   = int(parts[1]) if len(parts) > 1 and parts[1] != '??' else 6
            features['day']     = int(parts[2]) if len(parts) > 2 and parts[2] != '??' else 15
            features['decade']  = (year // 10) * 10 if year > 0 else 1950
            features['is_1900s']= 1 if 1900 <= year < 1920 else 0
            features['is_1920s']= 1 if 1920 <= year < 1940 else 0
            features['is_1940s']= 1 if 1940 <= year < 1960 else 0
            features['is_1960s']= 1 if 1960 <= year < 1980 else 0
            features['is_1980s']= 1 if 1980 <= year < 2000 else 0
            features['is_2000s']= 1 if year >= 2000 else 0
        else:
            features.update({'year':1950,'month':6,'day':15,'decade':1950,
                             'is_1900s':0,'is_1920s':0,'is_1940s':0,'is_1960s':0,'is_1980s':0,'is_2000s':0})
    except:
        features.update({'year':1950,'month':6,'day':15,'decade':1950,
                         'is_1900s':0,'is_1920s':0,'is_1940s':0,'is_1960s':0,'is_1980s':0,'is_2000s':0})

    eco = headers.get('ECO', 'A00')
    if eco and len(eco) >= 1:
        features['eco_letter'] = ord(eco[0]) - ord('A') if eco[0].isalpha() else 0
        features['eco_number'] = int(eco[1:]) if len(eco) > 1 and eco[1:].isdigit() else 0
    else:
        features['eco_letter'] = 0
        features['eco_number'] = 0

    result = headers.get('Result', '*')
    features['result_white_win'] = 1 if result == '1-0'     else 0
    features['result_black_win'] = 1 if result == '0-1'     else 0
    features['result_draw']      = 1 if result == '1/2-1/2' else 0
    return features


def extract_opening_features(moves, max_opening_moves=12):
    features = {}
    if len(moves) == 0:
        features.update({'opening_e4':0,'opening_d4':0,'opening_c4':0,'opening_nf3':0,
                         'opening_length':0,'early_knight_dev':0,'early_bishop_dev':0,
                         'early_castling':0,'early_queen_move':0,'opening_symmetry':0})
        return features

    opening_moves = moves[:max_opening_moves] if len(moves) >= max_opening_moves else moves
    opening_str   = ' '.join(opening_moves).lower()
    first_move    = opening_moves[0].lower() if opening_moves else ''

    features['opening_e4']  = 1 if 'e4' in first_move else 0
    features['opening_d4']  = 1 if 'd4' in first_move else 0
    features['opening_c4']  = 1 if 'c4' in first_move else 0
    features['opening_nf3'] = 1 if 'nf3' in first_move else 0
    features['opening_length']   = len(opening_moves)
    features['early_knight_dev'] = min(opening_str.count('n'), 4)
    features['early_bishop_dev'] = min(opening_str.count('b'), 4)
    features['early_castling']   = 1 if 'o-o' in opening_str else 0
    features['early_queen_move'] = 1 if 'q' in opening_str else 0
    features['opening_symmetry'] = 1 if len(opening_moves) >= 4 and opening_moves[0] == opening_moves[1] else 0
    return features


def extract_move_features(moves):
    features = {}
    board     = chess.Board()
    num_moves = len(moves)

    features['total_moves']        = num_moves
    features['game_length_short']  = 1 if num_moves < 30  else 0
    features['game_length_medium'] = 1 if 30 <= num_moves < 60 else 0
    features['game_length_long']   = 1 if num_moves >= 60 else 0

    captures = checks = castles = promotions = 0
    pawn_moves = knight_moves = bishop_moves = rook_moves = queen_moves = king_moves = 0
    center_moves = edge_moves = 0
    material_scores = []
    mobility_scores = []

    for move_san in moves:
        try:
            move = board.parse_san(move_san)
            if 'x' in move_san:           captures  += 1
            if '+' in move_san:           checks    += 1
            if '#' in move_san:           checks    += 2
            if 'O-O' in move_san or '0-0' in move_san: castles += 1
            if '=' in move_san:           promotions += 1

            piece = board.piece_at(move.from_square)
            if piece:
                pt = piece.piece_type
                if   pt == chess.PAWN:   pawn_moves   += 1
                elif pt == chess.KNIGHT: knight_moves  += 1
                elif pt == chess.BISHOP: bishop_moves  += 1
                elif pt == chess.ROOK:   rook_moves    += 1
                elif pt == chess.QUEEN:  queen_moves   += 1
                elif pt == chess.KING:   king_moves    += 1

            if move.to_square in [chess.D4, chess.E4, chess.D5, chess.E5]: center_moves += 1
            tf = chess.square_file(move.to_square)
            tr = chess.square_rank(move.to_square)
            if tf in [0, 7] or tr in [0, 7]: edge_moves += 1

            board.push(move)

            def pc(piece_type, color):
                return len(board.pieces(piece_type, color))

            mw = pc(chess.PAWN,chess.WHITE)*1 + pc(chess.KNIGHT,chess.WHITE)*3 + \
                 pc(chess.BISHOP,chess.WHITE)*3 + pc(chess.ROOK,chess.WHITE)*5 + pc(chess.QUEEN,chess.WHITE)*9
            mb = pc(chess.PAWN,chess.BLACK)*1 + pc(chess.KNIGHT,chess.BLACK)*3 + \
                 pc(chess.BISHOP,chess.BLACK)*3 + pc(chess.ROOK,chess.BLACK)*5 + pc(chess.QUEEN,chess.BLACK)*9
            material_scores.append(mw - mb)
            mobility_scores.append(board.legal_moves.count())
        except Exception:
            continue

    features.update({'captures':captures,'checks':checks,'castles':castles,'promotions':promotions,
                     'pawn_moves':pawn_moves,'knight_moves':knight_moves,'bishop_moves':bishop_moves,
                     'rook_moves':rook_moves,'queen_moves':queen_moves,'king_moves':king_moves,
                     'center_moves':center_moves,'edge_moves':edge_moves})

    if num_moves > 0:
        features.update({
            'capture_rate':   captures/num_moves, 'check_rate': checks/num_moves,
            'pawn_move_rate': pawn_moves/num_moves, 'knight_move_rate': knight_moves/num_moves,
            'bishop_move_rate': bishop_moves/num_moves, 'rook_move_rate': rook_moves/num_moves,
            'queen_move_rate':  queen_moves/num_moves,
            'piece_move_rate': (knight_moves+bishop_moves+rook_moves+queen_moves)/num_moves,
            'center_rate': center_moves/num_moves,
            'tactical_complexity': (captures+checks+promotions)/num_moves
        })
    else:
        features.update({'capture_rate':0,'check_rate':0,'pawn_move_rate':0,'knight_move_rate':0,
                         'bishop_move_rate':0,'rook_move_rate':0,'queen_move_rate':0,
                         'piece_move_rate':0,'center_rate':0,'tactical_complexity':0})

    if material_scores:
        features.update({'avg_material':np.mean(material_scores),'max_material':np.max(material_scores),
                         'min_material':np.min(material_scores),'material_variance':np.var(material_scores),
                         'material_range':np.max(material_scores)-np.min(material_scores)})
    else:
        features.update({'avg_material':0,'max_material':0,'min_material':0,'material_variance':0,'material_range':0})

    if mobility_scores:
        features.update({'avg_mobility':np.mean(mobility_scores),'max_mobility':np.max(mobility_scores),
                         'min_mobility':np.min(mobility_scores),'mobility_variance':np.var(mobility_scores)})
    else:
        features.update({'avg_mobility':0,'max_mobility':0,'min_mobility':0,'mobility_variance':0})
    return features


def extract_features_for_prediction(pgn_text):
    try:
        pgn  = io.StringIO(pgn_text)
        game = chess.pgn.read_game(pgn)
        if game is None:
            return None, None, None, "Could not parse PGN. Please check format."

        headers    = dict(game.headers)
        moves      = []
        temp_board = game.board()

        for move in game.mainline_moves():
            try:
                move_san = temp_board.san(move)
                moves.append(move_san)
                temp_board.push(move)
            except Exception:
                break

        if len(moves) < 10:
            return None, None, None, "Game is too short (< 10 moves). Please provide a longer game."

    except Exception as e:
        return None, None, None, f"PGN parsing error: {e}"

    white_name = headers.get('White', '').strip().lower()
    black_name = headers.get('Black', '').strip().lower()
    missing_indicators = {'', '?', 'unknown', 'missing', 'player', '____', '--'}

    is_white_missing = white_name in missing_indicators
    is_black_missing = black_name in missing_indicators

    if not is_white_missing:
        is_white_missing = any(ind in white_name for ind in ['unkn','player','___','--'])
    if not is_black_missing:
        is_black_missing = any(ind in black_name for ind in ['unkn','player','___','--'])

    if is_white_missing and is_black_missing:
        return None, None, None, "Both players marked as unknown. Please leave only one player unknown."
    if not is_white_missing and not is_black_missing:
        return None, None, None, (
            f"Both players ('{headers.get('White')}' vs '{headers.get('Black')}') appear known. "
            "Please set the unknown player to '?', 'unknown', or '--'."
        )

    missing_color = 'White' if is_white_missing else 'Black'

    features = OrderedDict()
    features['is_white'] = 1 if missing_color == 'White' else 0
    features['is_black'] = 1 if missing_color == 'Black' else 0
    features.update(extract_metadata_features(headers))
    features.update(extract_opening_features(moves))
    features.update(extract_move_features(moves))

    return features, missing_color, moves, None


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained model from disk. Searches common locations."""
    search_paths = [
        "chess_player_classifier_optimized.pkl",
        os.path.join(os.path.dirname(__file__),
                     "MLProj_Chess Player Classifier",
                     "Chess Player Style Classification",
                     "All Files at a Place to Run Code",
                     "chess_player_classifier_optimized.pkl"),
    ]
    for path in search_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f), path
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER METADATA
# ─────────────────────────────────────────────────────────────────────────────

PLAYER_INFO = {
    "Marshall": {
        "emoji": "⚔️",
        "era": "1875 – 1944",
        "style": "Tactical Genius",
        "desc": "Famous for the Marshall Attack and bold sacrificial play. One of the first American chess stars.",
        "color": "#ff6b6b",
        "flag": "🇺🇸",
    },
    "Li": {
        "emoji": "🐉",
        "era": "1989 – present",
        "style": "Positional Precision",
        "desc": "Chinese grandmaster known for deep preparation and solid positional understanding.",
        "color": "#4f9eff",
        "flag": "🇨🇳",
    },
    "Tartakower": {
        "emoji": "🎭",
        "era": "1887 – 1956",
        "style": "Hypermodern Pioneer",
        "desc": "Hypermodern theoretician and witty chess writer. Author of many famous chess aphorisms.",
        "color": "#a78bfa",
        "flag": "🇫🇷",
    },
    "Najdorf": {
        "emoji": "🦁",
        "era": "1910 – 1997",
        "style": "Aggressive Attacker",
        "desc": "Namesake of the Najdorf Sicilian. One of the most combative grandmasters in history.",
        "color": "#f59e0b",
        "flag": "🇦🇷",
    },
    "Kudrin": {
        "emoji": "♜",
        "era": "1959 – present",
        "style": "Solid Strategist",
        "desc": "American grandmaster known for solid, reliable play and deep endgame technique.",
        "color": "#00d4aa",
        "flag": "🇺🇸",
    },
}

EXAMPLE_PGN = """[Event "World Championship Match"]
[Site "New York, NY USA"]
[Date "1907.04.23"]
[White "?"]
[Black "Tarrasch, S"]
[Result "1-0"]
[ECO "D31"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 Nbd7 6. Nf3 O-O 7. Rc1 c6
8. Bd3 dxc4 9. Bxc4 Nd5 10. Bxe7 Qxe7 11. O-O Nxc3 12. Rxc3 e5 13. dxe5
Nxe5 14. Nxe5 Qxe5 15. f4 Qe7 16. f5 Rd8 17. Qe2 Bd7 18. Bd3 Bc8 19. e4
b5 20. e5 b4 21. Rc2 c5 22. Be4 Ra7 23. f6 gxf6 24. exf6 Qd6 25. Rcf2 Be6
26. Rxf7 Rxf7 27. Rxf7 Kxf7 28. Qh5+ Ke7 29. Qxh7+ Kd8 30. Qg8+ Kc7
31. Qe8 Bd7 32. Qxd7+ Kxd7 33. Bf5+ Ke7 34. Bxd7 1-0"""


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def render_probability_bars(players, probabilities, predicted_idx, player_colors):
    bar_html = ""
    sorted_idx = np.argsort(probabilities)[::-1]
    colors = [
        "linear-gradient(90deg,#00d4aa,#4f9eff)",
        "linear-gradient(90deg,#4f9eff,#a78bfa)",
        "linear-gradient(90deg,#a78bfa,#f59e0b)",
        "linear-gradient(90deg,#f59e0b,#ff6b6b)",
        "linear-gradient(90deg,#ff6b6b,#ff9999)",
    ]
    for rank, idx in enumerate(sorted_idx):
        player = players[idx]
        prob   = probabilities[idx] * 100
        is_top = (idx == predicted_idx)
        fill   = colors[rank % len(colors)]
        star   = " ★" if is_top else ""
        bar_html += f"""
        <div class="prob-row">
            <div class="prob-label" style="{'color:#f0c040;font-weight:700;' if is_top else ''}">{player}{star}</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{prob:.1f}%;background:{fill};"></div>
            </div>
            <div class="prob-value">{prob:.1f}%</div>
        </div>"""
    return bar_html


def render_game_stats(moves, features):
    total = len(moves)
    captures        = features.get('captures', 0)
    checks          = features.get('checks', 0)
    center_moves    = features.get('center_moves', 0)
    tactical_cmplx  = features.get('tactical_complexity', 0)

    stats = [
        ("Total Moves", total, ""),
        ("Captures", captures, ""),
        ("Checks", checks, ""),
        ("Center Control", center_moves, ""),
        ("Tactical Complexity", f"{tactical_cmplx:.2f}", ""),
        ("Avg Mobility", f"{features.get('avg_mobility',0):.1f}", ""),
    ]
    cols = st.columns(len(stats))
    for col, (label, val, _) in zip(cols, stats):
        col.markdown(f"""
        <div class="stat-chip">
            <div class="stat-value">{val}</div>
            <div class="stat-label">{label}</div>
        </div>""", unsafe_allow_html=True)


def make_radar_chart(features, feature_names):
    """Create a radar chart of key game features."""
    radar_features = {
        'Capture Rate':      features.get('capture_rate', 0),
        'Center Control':    features.get('center_rate', 0),
        'Piece Activity':    features.get('piece_move_rate', 0),
        'Tactical Density':  features.get('tactical_complexity', 0),
        'Avg Mobility':      features.get('avg_mobility', 0) / 40,  # normalize
        'Check Aggression':  features.get('check_rate', 0),
    }
    labels = list(radar_features.keys())
    values = list(radar_features.values())
    values_norm = [min(v, 1.0) for v in values]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_norm + [values_norm[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(0,212,170,0.15)',
        line=dict(color='#00d4aa', width=2),
        name='Game Profile',
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(26,31,46,0.8)',
            radialaxis=dict(visible=True, range=[0, 1],
                            tickfont=dict(color='#8892a4', size=9),
                            gridcolor='rgba(255,255,255,0.08)'),
            angularaxis=dict(tickfont=dict(color='#f0f2f8', size=11),
                             gridcolor='rgba(255,255,255,0.08)'),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Outfit', color='#f0f2f8'),
        showlegend=False,
        margin=dict(l=60, r=60, t=30, b=30),
        height=340,
    )
    return fig


def make_probability_donut(players, probabilities):
    colors = ['#00d4aa','#4f9eff','#a78bfa','#f59e0b','#ff6b6b']
    fig = go.Figure(go.Pie(
        labels=players,
        values=[p*100 for p in probabilities],
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='#0d0f14', width=2)),
        textinfo='label+percent',
        textfont=dict(family='Outfit', size=11, color='#f0f2f8'),
        hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>',
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Outfit', color='#f0f2f8'),
        showlegend=True,
        legend=dict(font=dict(color='#f0f2f8', size=11),
                    bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=20, r=20, t=20, b=20),
        height=320,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(model_data):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1.5rem 0 1rem;">
            <div style="font-size:2.8rem;">♟️</div>
            <div style="font-size:1.1rem;font-weight:700;color:#f0c040;letter-spacing:1px;">Chess GrandMaster</div>
            <div style="font-size:0.78rem;color:#8892a4;letter-spacing:2px;text-transform:uppercase;">Classifier</div>
        </div>
        <hr style="border-color:rgba(240,192,64,0.15);margin:0 0 1.5rem;">
        """, unsafe_allow_html=True)

        if model_data:
            acc = model_data.get('test_accuracy', 0) * 100
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:1.2rem;">
                <div style="font-size:0.7rem;color:#8892a4;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:0.5rem;">Model Accuracy</div>
                <div style="font-size:2.2rem;font-weight:800;color:#00d4aa;">{acc:.1f}%</div>
                <div style="font-size:0.72rem;color:#8892a4;margin-top:0.3rem;">MLP Neural Network · RFECV Optimized</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div style="font-size:0.75rem;font-weight:600;color:#8892a4;text-transform:uppercase;letter-spacing:1.5px;margin:1rem 0 0.8rem;">Classifiable Players</div>', unsafe_allow_html=True)

        for name, info in PLAYER_INFO.items():
            st.markdown(f"""
            <div class="player-card">
                <div style="display:flex;align-items:center;gap:0.6rem;">
                    <span style="font-size:1.3rem;">{info['emoji']}</span>
                    <div>
                        <div class="player-name">{info['flag']} {name}</div>
                        <div class="player-era">{info['era']} · {info['style']}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <hr style="border-color:rgba(255,255,255,0.07);margin:1.5rem 0 1rem;">
        <div style="font-size:0.75rem;font-weight:600;color:#8892a4;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:0.8rem;">How It Works</div>
        <div style="font-size:0.8rem;color:#8892a4;line-height:1.7;">
            1️⃣ <b style="color:#f0f2f8;">Paste PGN</b> with one unknown player<br>
            2️⃣ <b style="color:#f0f2f8;">Features extracted</b> from moves, openings & metadata<br>
            3️⃣ <b style="color:#f0f2f8;">MLP Neural Network</b> classifies the style<br>
            4️⃣ <b style="color:#f0f2f8;">Confidence scores</b> shown for all players
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Load model
    model_data, model_path = load_model()

    # Sidebar
    render_sidebar(model_data)

    # ── Hero Banner ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <div class="chess-pieces-row">♔ ♕ ♖ ♗ ♘ ♙</div>
        <div class="hero-title">Chess GrandMaster Classifier</div>
        <div class="hero-subtitle">
            Paste a PGN game with one unknown player — our AI will identify the grandmaster by their playing style
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model status alert ───────────────────────────────────────────────────
    if model_data is None:
        st.error(
            "⚠️  **Model not found.** Please place `chess_player_classifier_optimized.pkl` "
            "in the same directory as `app.py` and restart."
        )
        st.info(
            "Expected file: `chess_player_classifier_optimized.pkl`\n\n"
            "Run `Feature_Extraction_Training_Testing.py` to generate it."
        )
        return

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_classify, tab_about, tab_plots = st.tabs(["🔮  Classify Game", "📖  About Players", "📊  Model Insights"])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1 — Classify
    # ═══════════════════════════════════════════════════════════════════════
    with tab_classify:
        col_input, col_result = st.columns([1, 1], gap="large")

        with col_input:
            st.markdown('<div class="section-header">📋 PGN Input</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="card" style="padding:1.2rem 1.5rem;">
                <div class="card-title">ℹ️ Instructions</div>
                <div style="font-size:0.85rem;color:#8892a4;line-height:1.8;">
                    Set the <b style="color:#f0c040;">unknown player</b> to <code style="background:#1a1f2e;padding:2px 6px;border-radius:4px;">?</code>,
                    <code style="background:#1a1f2e;padding:2px 6px;border-radius:4px;">unknown</code>, or
                    <code style="background:#1a1f2e;padding:2px 6px;border-radius:4px;">--</code>
                    in the White or Black header field.
                </div>
            </div>
            """, unsafe_allow_html=True)

            pgn_input = st.text_area(
                "Paste PGN here",
                height=320,
                placeholder='[Event "..."]\n[White "?"]\n[Black "Opponent"]\n...\n\n1. e4 e5 2. Nf3 ...',
                label_visibility="collapsed",
                key="pgn_input_area"
            )

            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                classify_btn = st.button("♟️  Classify Player", key="classify_btn")
            with col_btn2:
                example_btn  = st.button("📄  Load Example", key="example_btn")

            if example_btn:
                st.session_state['pgn_example'] = EXAMPLE_PGN
                st.rerun()

            # Use example if loaded
            if 'pgn_example' in st.session_state and not pgn_input:
                pgn_input = st.session_state['pgn_example']

        with col_result:
            st.markdown('<div class="section-header">🎯 Prediction</div>', unsafe_allow_html=True)

            result_placeholder = st.empty()

            if not classify_btn and 'last_result' not in st.session_state:
                result_placeholder.markdown("""
                <div class="card" style="text-align:center;padding:3rem 2rem;min-height:280px;
                     display:flex;flex-direction:column;align-items:center;justify-content:center;">
                    <div style="font-size:3.5rem;opacity:0.3;margin-bottom:1rem;">♟️</div>
                    <div style="color:#8892a4;font-size:0.95rem;">
                        Paste a PGN and click <b style="color:#f0c040;">Classify Player</b> to begin
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Run classification ──────────────────────────────────────────────
        if classify_btn:
            if not pgn_input or pgn_input.strip() == '':
                with col_result:
                    st.warning("Please paste a PGN game first.")
            else:
                with col_result:
                    with st.spinner("🔍 Analysing game..."):
                        features, missing_color, moves, error = extract_features_for_prediction(pgn_input)

                    if error:
                        st.error(f"**Extraction failed:** {error}")
                    else:
                        # Build feature vector
                        model       = model_data['model']
                        scaler      = model_data['scaler']
                        le          = model_data['label_encoder']
                        feat_names  = model_data['feature_names']
                        players     = list(le.classes_)

                        feat_vec = np.array([[features.get(f, 0) for f in feat_names]])
                        X_scaled = scaler.transform(feat_vec)

                        pred_idx   = model.predict(X_scaled)[0]
                        probs      = model.predict_proba(X_scaled)[0]
                        pred_name  = le.inverse_transform([pred_idx])[0]
                        confidence = probs[pred_idx] * 100
                        info       = PLAYER_INFO.get(pred_name, {})

                        # Store result
                        st.session_state['last_result'] = {
                            'pred_name': pred_name, 'missing_color': missing_color,
                            'confidence': confidence, 'probs': probs,
                            'players': players, 'pred_idx': pred_idx,
                            'features': features, 'moves': moves,
                        }

                        badge_style = "background:#f0f2f8;color:#0d0f14;" if missing_color == 'White' else "background:#2a2a2a;color:#f0f2f8;"
                        bar_html = render_probability_bars(players, probs, pred_idx,
                                                           {p: PLAYER_INFO.get(p,{}).get('color','#888') for p in players})

                        result_placeholder.markdown(f"""
                        <div class="result-box">
                            <div style="font-size:0.85rem;color:#8892a4;margin-bottom:0.3rem;">PREDICTED PLAYER</div>
                            <div style="font-size:2.5rem;margin:0.2rem 0;">{info.get('emoji','♟️')}</div>
                            <div class="result-player">{pred_name}</div>
                            <span class="result-color-badge" style="{badge_style}">
                                Playing as {missing_color}
                            </span>
                            <div class="result-confidence">
                                {confidence:.1f}% Confidence
                            </div>
                            <div style="font-size:0.82rem;color:#8892a4;margin:0.8rem 0 1.5rem;font-style:italic;">
                                "{info.get('desc','')}"
                            </div>
                            <div style="text-align:left;margin-top:1rem;">
                                <div style="font-size:0.72rem;color:#8892a4;letter-spacing:1.5px;
                                     text-transform:uppercase;margin-bottom:0.8rem;">Confidence Breakdown</div>
                                {bar_html}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        # ── Game stats & charts (shown below both columns) ──────────────────
        if 'last_result' in st.session_state:
            res = st.session_state['last_result']
            st.markdown('<div class="section-header" style="margin-top:2.5rem;">📈 Game Analysis</div>', unsafe_allow_html=True)
            render_game_stats(res['moves'], res['features'])

            st.markdown("<br>", unsafe_allow_html=True)
            col_radar, col_donut = st.columns(2)
            with col_radar:
                st.markdown('<div class="card-title" style="color:#f0c040;font-size:0.82rem;letter-spacing:1.5px;">GAME STYLE RADAR</div>', unsafe_allow_html=True)
                st.plotly_chart(make_radar_chart(res['features'], model_data['feature_names']),
                                use_container_width=True, key="radar")
            with col_donut:
                st.markdown('<div class="card-title" style="color:#f0c040;font-size:0.82rem;letter-spacing:1.5px;">PROBABILITY DISTRIBUTION</div>', unsafe_allow_html=True)
                st.plotly_chart(make_probability_donut(res['players'], res['probs']),
                                use_container_width=True, key="donut")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2 — About Players
    # ═══════════════════════════════════════════════════════════════════════
    with tab_about:
        st.markdown('<div class="section-header">👑 The Grandmasters</div>', unsafe_allow_html=True)

        cols = st.columns(len(PLAYER_INFO))
        for col, (name, info) in zip(cols, PLAYER_INFO.items()):
            with col:
                st.markdown(f"""
                <div class="card" style="text-align:center;border-top:3px solid {info['color']};">
                    <div style="font-size:2.5rem;margin-bottom:0.5rem;">{info['emoji']}</div>
                    <div style="font-size:1.1rem;font-weight:700;color:{info['color']};">{info['flag']} {name}</div>
                    <div style="font-size:0.75rem;color:#8892a4;margin:0.3rem 0 0.8rem;
                         text-transform:uppercase;letter-spacing:1px;">{info['style']}</div>
                    <div style="font-size:0.8rem;color:#8892a4;line-height:1.6;">{info['desc']}</div>
                    <div style="font-size:0.72rem;color:{info['color']};margin-top:1rem;
                         font-weight:600;">{info['era']}</div>
                </div>
                """, unsafe_allow_html=True)

        # Model features table
        st.markdown('<div class="section-header" style="margin-top:2rem;">🔬 Features Used by the Model</div>', unsafe_allow_html=True)
        if model_data:
            feat_names = model_data['feature_names']
            st.markdown(f"The model uses **{len(feat_names)} optimally selected features** (via RFECV):", unsafe_allow_html=True)
            badge_html = "".join(f'<span class="feat-badge">{f}</span>' for f in feat_names)
            st.markdown(f'<div style="margin-top:0.5rem;">{badge_html}</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3 — Model Insights (evaluation plots)
    # ═══════════════════════════════════════════════════════════════════════
    with tab_plots:
        st.markdown('<div class="section-header">📊 Model Evaluation Plots</div>', unsafe_allow_html=True)

        # plots/ folder lives next to app.py — committed to the repo
        plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

        plot_files = {
            "Confusion Matrix":       "confusion_matrix.png",
            "ROC Curve":              "roc_curve.png",
            "PCA 2D Projection":      "pca_2d_projection.png",
            "Class Distribution":     "class_distribution.png",
            "RFECV Optimization":     "rfecv_optimization_curve.png",
            "Classification Report": "Classification Report.png",
            "Metrics Table":          "Metrices Table.png",
        }

        found = {}
        for title, fname in plot_files.items():
            full_path = os.path.join(plots_dir, fname)
            if os.path.isfile(full_path):
                found[title] = full_path

        if not found:
            st.info(
                "📂 Evaluation plots not found.\n\n"
                f"Expected a `plots/` folder next to `app.py` containing the PNG files.\n\n"
                "Make sure the `plots/` folder is committed to your GitHub repository."
            )
        else:
            # Display in a 2-column grid
            items = list(found.items())
            for i in range(0, len(items), 2):
                row_items = items[i:i+2]
                cols = st.columns(len(row_items))
                for col, (title, path) in zip(cols, row_items):
                    with col:
                        st.markdown(f"""
                        <div class="card" style="padding:1rem;">
                            <div class="card-title">📈 {title}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        try:
                            with open(path, "rb") as img_f:
                                img_bytes = img_f.read()
                            if len(img_bytes) > 0:
                                st.image(img_bytes, use_container_width=True)
                            else:
                                st.warning(f"⚠️ {title}: image file is empty.")
                        except Exception as img_err:
                            st.warning(f"⚠️ Could not load **{title}**: {img_err}")

        # Model info
        if model_data:
            st.markdown('<div class="section-header" style="margin-top:2rem;">🤖 Model Architecture</div>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""<div class="stat-chip"><div class="stat-value">{model_data.get('test_accuracy',0)*100:.1f}%</div><div class="stat-label">Test Accuracy</div></div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class="stat-chip"><div class="stat-value">{len(model_data.get('feature_names',[]))}</div><div class="stat-label">Features</div></div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""<div class="stat-chip"><div class="stat-value">5</div><div class="stat-label">Players</div></div>""", unsafe_allow_html=True)
            with col4:
                st.markdown(f"""<div class="stat-chip"><div class="stat-value">MLP</div><div class="stat-label">Algorithm</div></div>""", unsafe_allow_html=True)

            st.markdown("""
            <div class="card" style="margin-top:1.5rem;">
                <div class="card-title">🏗️ Architecture Details</div>
                <div style="font-size:0.88rem;color:#8892a4;line-height:2;">
                    <b style="color:#f0f2f8;">Type:</b> Multi-Layer Perceptron (MLPClassifier) · sklearn<br>
                    <b style="color:#f0f2f8;">Hidden Layers:</b> 256 → 128 → 64 → 32 neurons (ReLU activation)<br>
                    <b style="color:#f0f2f8;">Optimizer:</b> Adam · LR = 0.001 · Adaptive schedule<br>
                    <b style="color:#f0f2f8;">Regularization:</b> L2 α=0.001 · Early stopping (20 patience)<br>
                    <b style="color:#f0f2f8;">Feature Selection:</b> RFECV with Logistic Regression surrogate (5-fold CV)<br>
                    <b style="color:#f0f2f8;">Features:</b> Metadata + Opening patterns + Move statistics
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:3rem 0 1rem;color:#8892a4;font-size:0.78rem;letter-spacing:0.5px;">
        ♟️ Chess GrandMaster Classifier &nbsp;·&nbsp; ML Project &nbsp;·&nbsp;
        Built with Streamlit &nbsp;·&nbsp; MLP Neural Network
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
