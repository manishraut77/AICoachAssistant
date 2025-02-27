import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from statsbombpy import sb

# Constants
FIELD_LENGTH = 105
FIELD_WIDTH = 68

# Helper function used in data processing
def convert_coords(sb_x, sb_y):
    """Convert StatsBomb coordinates (120x80) to dashboard coordinates (105x68)."""
    return (sb_x / 120 * FIELD_LENGTH, sb_y / 80 * FIELD_WIDTH)

# -------------------------
# StatsBomb Data Acquisition
# -------------------------
events = sb.events(match_id=3869685)

# Extract shot data for xG model training
shots_df = events[events['type'] == 'Shot'].copy()
shots_df['distance'] = np.sqrt((120 - shots_df['location'].apply(lambda x: x[0])) ** 2 +
                               (40 - shots_df['location'].apply(lambda x: x[1])) ** 2)
shots_df['angle'] = np.arctan2(40 - shots_df['location'].apply(lambda x: x[1]),
                               120 - shots_df['location'].apply(lambda x: x[0])).abs() * 180 / np.pi
shots_df['goal'] = shots_df['shot_outcome'].apply(lambda x: 1 if x == 'Goal' else 0)

# Train a simple xG model
X = shots_df[['distance', 'angle']]
y = shots_df['goal']
xg_model = LogisticRegression().fit(X, y)
shots_df['xG'] = xg_model.predict_proba(X)[:, 1]

# Convert shot locations to dashboard coordinates (105x68)
shots_df['dashboard_x'] = shots_df['location'].apply(lambda loc: convert_coords(loc[0], loc[1])[0])
shots_df['dashboard_y'] = shots_df['location'].apply(lambda loc: convert_coords(loc[0], loc[1])[1])

# Compute optimal positions for each team based on high xG shots
argentina_high_xg_shots = shots_df[(shots_df['team'] == 'Argentina') & (shots_df['xG'] > 0.3)]
if not argentina_high_xg_shots.empty:
    argentina_optimal_x = argentina_high_xg_shots['dashboard_x'].mean()
    argentina_optimal_y = argentina_high_xg_shots['dashboard_y'].mean()
else:
    argentina_optimal_x, argentina_optimal_y = 90, 34  # Default near France's penalty area for Argentina

france_high_xg_shots = shots_df[(shots_df['team'] == 'France') & (shots_df['xG'] > 0.3)]
if not france_high_xg_shots.empty:
    france_optimal_x = france_high_xg_shots['dashboard_x'].mean()
    france_optimal_y = france_high_xg_shots['dashboard_y'].mean()
else:
    france_optimal_x, france_optimal_y = 10, 34  # Default near Argentina's penalty area for France

# Team statistics for tactical decisions
teams_df = pd.DataFrame({
    'Team': ['Argentina', 'France'],
    'Offensive_xG': [shots_df[shots_df['team'] == 'Argentina']['xG'].sum(),
                     shots_df[shots_df['team'] == 'France']['xG'].sum()],
    'Defensive_xG': [shots_df[shots_df['team'] == 'France']['xG'].sum(),
                     shots_df[shots_df['team'] == 'Argentina']['xG'].sum()]
})
teams_df['xG_Differential'] = teams_df['Offensive_xG'] - teams_df['Defensive_xG']

# -------------------------
# Player Data and Initial Positions
# -------------------------
# Argentina players and initial positions
argentina_players = [
    "Emiliano Martinez", "Nahuel Molina", "Cristian Romero", "Nicolas Otamendi",
    "Nicolas Tagliafico", "Rodrigo De Paul", "Leandro Paredes", "Alexis Mac Allister",
    "Angel Di Maria", "Lionel Messi", "Julian Alvarez"
]

argentina_positions = {
    "Emiliano Martinez": (5, 34),      # GK
    "Nahuel Molina": (20, 10),         # RB
    "Cristian Romero": (20, 30),       # CB
    "Nicolas Otamendi": (20, 50),      # CB
    "Nicolas Tagliafico": (20, 70),    # LB
    "Rodrigo De Paul": (40, 20),       # RM
    "Leandro Paredes": (45, 34),       # CM
    "Alexis Mac Allister": (40, 50),   # LM
    "Angel Di Maria": (60, 20),        # RW
    "Lionel Messi": (65, 34),          # AM
    "Julian Alvarez": (70, 50)         # ST
}

# France players and initial positions
france_players = [
    "Hugo Lloris", "Jules Koundé", "Raphaël Varane", "Dayot Upamecano", "Theo Hernández",
    "Aurélien Tchouaméni", "Adrien Rabiot", "Ousmane Dembélé", "Antoine Griezmann",
    "Kylian Mbappé", "Olivier Giroud"
]

france_positions = {
    "Hugo Lloris": (100, 34),         # GK
    "Jules Koundé": (85, 60),         # RB
    "Raphaël Varane": (85, 40),       # CB
    "Dayot Upamecano": (85, 28),      # CB
    "Theo Hernández": (85, 8),        # LB
    "Aurélien Tchouaméni": (75, 34), # DM
    "Adrien Rabiot": (70, 34),        # CM
    "Ousmane Dembélé": (60, 60),      # RW
    "Antoine Griezmann": (60, 34),    # AM
    "Kylian Mbappé": (60, 8),         # LW
    "Olivier Giroud": (55, 34)        # ST
}

player_positions = {**argentina_positions, **france_positions}

# Player stats for tactical adjustments (example data)
player_past_stats = {
    "Lionel Messi": {"matches": 6, "goals": 5, "assists": 3, "passes": 320, "avg_rating": 8.4},
    "Julian Alvarez": {"matches": 6, "goals": 4, "assists": 0, "passes": 180, "avg_rating": 7.9},
    "Angel Di Maria": {"matches": 4, "goals": 1, "assists": 1, "passes": 120, "avg_rating": 7.5},
    "Rodrigo De Paul": {"matches": 6, "goals": 0, "assists": 1, "passes": 290, "avg_rating": 7.6},
    "Leandro Paredes": {"matches": 5, "goals": 0, "assists": 0, "passes": 200, "avg_rating": 7.3},
    "Alexis Mac Allister": {"matches": 6, "goals": 1, "assists": 1, "passes": 250, "avg_rating": 7.7},
    "Emiliano Martinez": {"matches": 6, "goals": 0, "assists": 0, "passes": 50, "avg_rating": 7.2},
    "Cristian Romero": {"matches": 6, "goals": 0, "assists": 0, "passes": 120, "avg_rating": 7.5},
    "Nicolas Otamendi": {"matches": 6, "goals": 0, "assists": 0, "passes": 180, "avg_rating": 7.4},
    "Nicolas Tagliafico": {"matches": 5, "goals": 0, "assists": 0, "passes": 150, "avg_rating": 7.3},
    "Nahuel Molina": {"matches": 6, "goals": 1, "assists": 1, "passes": 170, "avg_rating": 7.4}
}

# Map player IDs to names from lineups
lineups = sb.lineups(match_id=3869685)
player_id_to_name = {}
for team in lineups:
    for _, row in lineups[team].iterrows():
        player_id_to_name[row['player_id']] = row['player_name']

# Define player roles
player_roles = {
    "Emiliano Martinez": "GK",
    "Nahuel Molina": "DEF",
    "Cristian Romero": "DEF",
    "Nicolas Otamendi": "DEF",
    "Nicolas Tagliafico": "DEF",
    "Rodrigo De Paul": "MID",
    "Leandro Paredes": "MID",
    "Alexis Mac Allister": "MID",
    "Angel Di Maria": "FWD",
    "Lionel Messi": "FWD",
    "Julian Alvarez": "FWD",
    "Hugo Lloris": "GK",
    "Jules Koundé": "DEF",
    "Raphaël Varane": "DEF",
    "Dayot Upamecano": "DEF",
    "Theo Hernández": "DEF",
    "Aurélien Tchouaméni": "MID",
    "Adrien Rabiot": "MID",
    "Ousmane Dembélé": "MID",
    "Antoine Griezmann": "MID",
    "Kylian Mbappé": "FWD",
    "Olivier Giroud": "FWD"
}

# -------------------------
# Helper Functions
# -------------------------
def get_player_bounds(player):
    """Return movement bounds (x_min, x_max, y_min, y_max) based on player role and team."""
    role = player_roles[player]
    team = 'Argentina' if player in argentina_players else 'France'
    if role == 'GK':
        if team == 'Argentina':
            return (0, 18, 22, 46)  # Near goal post
        else:
            return (87, 105, 22, 46)
    elif role == 'DEF':
        if team == 'Argentina':
            return (0, 52.5, 0, 68)  # Defensive half
        else:
            return (52.5, 105, 0, 68)
    elif role == 'MID':
        return (0, 105, 0, 68)  # Full field
    elif role == 'FWD':
        if team == 'Argentina':
            return (52.5, 105, 0, 68)  # Attacking half
        else:
            return (0, 52.5, 0, 68)

def update_player_positions(n):
    """Update player positions smoothly with occasional aggressive movement."""
    current_event = events.iloc[n % len(events)]
    smoothing_factor = 0.3  # Adjust for smoother transitions (0 to 1, lower = smoother)
    aggressive_chance = 0.2  # 20% chance for aggressive movement

    # Players to update based on event
    players_to_set = []
    if 'player_id' in current_event and current_event['player_id'] in player_id_to_name:
        player_name = player_id_to_name[current_event['player_id']]
        if player_name in player_positions and 'location' in current_event:
            sb_x, sb_y = current_event['location']
            target_x, target_y = convert_coords(sb_x, sb_y)
            x, y = player_positions[player_name]
            # Smoothly interpolate toward event location
            new_x = x + smoothing_factor * (target_x - x)
            new_y = y + smoothing_factor * (target_y - y)
            player_positions[player_name] = (new_x, new_y)
            players_to_set.append(player_name)

    if current_event['type'] == 'Pass' and 'pass_recipient_id' in current_event:
        recipient_id = current_event['pass_recipient_id']
        if recipient_id in player_id_to_name:
            recipient_name = player_id_to_name[recipient_id]
            if recipient_name in player_positions and 'pass_end_location' in current_event:
                sb_x, sb_y = current_event['pass_end_location']
                target_x, target_y = convert_coords(sb_x, sb_y)
                x, y = player_positions[recipient_name]
                # Smoothly interpolate toward pass end location
                new_x = x + smoothing_factor * (target_x - x)
                new_y = y + smoothing_factor * (target_y - y)
                player_positions[recipient_name] = (new_x, new_y)
                players_to_set.append(recipient_name)

    # Add smooth random movement for other players with occasional aggressive bursts
    for player in player_positions:
        if player not in players_to_set:
            x, y = player_positions[player]
            role = player_roles[player]
            # Occasionally increase movement for non-GK players to simulate aggression
            if role != 'GK' and random.random() < aggressive_chance:
                dx = random.uniform(-2.0, 2.0)  # Larger, aggressive movement
                dy = random.uniform(-2.0, 2.0)
            else:
                # Normal smooth movement
                dx = random.uniform(-0.2, 0.2) if role == 'GK' else random.uniform(-0.5, 0.5)
                dy = random.uniform(-0.2, 0.2) if role == 'GK' else random.uniform(-0.5, 0.5)
            target_x = x + dx
            target_y = y + dy
            # Smoothly interpolate toward random target
            new_x = x + smoothing_factor * (target_x - x)
            new_y = y + smoothing_factor * (target_y - y)
            # Clip to role-specific bounds
            x_min, x_max, y_min, y_max = get_player_bounds(player)
            new_x = max(x_min, min(new_x, x_max))
            new_y = max(y_min, min(new_y, y_max))
            player_positions[player] = (new_x, new_y)

def ai_tactic_and_position(player, team):
    """Suggest detailed tactics and optimal positions based on player role, team situation, and stats."""
    # Extract player role and data
    role = player_roles[player]
    player_data = player_past_stats.get(player, {})
    team_data = teams_df[teams_df['Team'] == team].iloc[0]
    x, y = player_positions[player]  # Current position

    # Determine optimal position based on role and team
    if role == 'FWD':
        if team == 'Argentina':
            optimal_x, optimal_y = argentina_optimal_x, argentina_optimal_y
        else:
            optimal_x, optimal_y = france_optimal_x, france_optimal_y
    elif role == 'DEF':
        if team == 'Argentina':
            optimal_x, optimal_y = france_optimal_x, france_optimal_y  # Defend opponent's high xG area
        else:
            optimal_x, optimal_y = argentina_optimal_x, argentina_optimal_y
    elif role == 'MID':
        if team == 'Argentina':
            optimal_x = (argentina_optimal_x + france_optimal_x) / 2
            optimal_y = (argentina_optimal_y + france_optimal_y) / 2
        else:
            optimal_x = (france_optimal_x + argentina_optimal_x) / 2
            optimal_y = (france_optimal_y + argentina_optimal_y) / 2
    elif role == 'GK':
        if team == 'Argentina':
            optimal_x, optimal_y = 5, 34  # Near Argentina's goal
        else:
            optimal_x, optimal_y = 100, 34  # Near France's goal

    # Role-based tactical suggestions (10+ per role)
    if role == 'FWD':
        if team_data['xG_Differential'] > 0.5:
            if x < optimal_x - 10:
                tactic = f"Surge forward to ({optimal_x:.1f}, {optimal_y:.1f}). Your team’s lead (xG diff {team_data['xG_Differential']:.2f}) opens gaps—exploit them with pace."
            elif abs(y - 34) > 15:
                tactic = f"Cut inside to ({optimal_x:.1f}, {optimal_y:.1f}). Dominance (xG diff {team_data['xG_Differential']:.2f}) lets you attack centrally—unleash a shot."
            elif x > 85:
                tactic = f"Hold position near ({optimal_x:.1f}, {optimal_y:.1f}). With xG lead ({team_data['xG_Differential']:.2f}), draw defenders out for teammates."
            elif random.random() < 0.3:
                tactic = f"Drop back slightly from ({x:.1f}, {y:.1f}) to link play. Your edge (xG diff {team_data['xG_Differential']:.2f}) allows space creation."
            else:
                tactic = f"Push to ({optimal_x:.1f}, {optimal_y:.1f}). Team’s dominance (xG diff {team_data['xG_Differential']:.2f})—finish clinically in the box."
        else:
            if x < optimal_x - 10:
                tactic = f"Advance cautiously to ({optimal_x:.1f}, {optimal_y:.1f}). Tight game (xG diff {team_data['xG_Differential']:.2f})—wait for openings."
            elif abs(y - 34) > 15:
                tactic = f"Drift to ({optimal_x:.1f}, {optimal_y:.1f}). Close match (xG diff {team_data['xG_Differential']:.2f})—exploit wide gaps."
            elif x > 85:
                tactic = f"Stay near ({optimal_x:.1f}, {optimal_y:.1f}). Even contest (xG diff {team_data['xG_Differential']:.2f})—hold for a counter."
            elif random.random() < 0.3:
                tactic = f"Track back from ({x:.1f}, {y:.1f}) to support. Tight (xG diff {team_data['xG_Differential']:.2f})—help midfield."
            else:
                tactic = f"Move to ({optimal_x:.1f}, {optimal_y:.1f}). Balanced game (xG diff {team_data['xG_Differential']:.2f})—strike when ready."

    elif role == 'DEF':
        if team_data['xG_Differential'] > 0.5:
            if x > optimal_x + 10:
                tactic = f"Push up to ({optimal_x:.1f}, {optimal_y:.1f}). Lead (xG diff {team_data['xG_Differential']:.2f})—press their forwards high."
            elif abs(y - optimal_y) > 10:
                tactic = f"Shift to ({optimal_x:.1f}, {optimal_y:.1f}). Advantage (xG diff {team_data['xG_Differential']:.2f})—cover wide threats."
            elif x < 20:
                tactic = f"Hold at ({x:.1f}, {y:.1f}). Strong xG ({team_data['xG_Differential']:.2f})—block counters early."
            elif random.random() < 0.3:
                tactic = f"Step up from ({x:.1f}, {y:.1f}) to intercept. Lead (xG diff {team_data['xG_Differential']:.2f})—disrupt their rhythm."
            else:
                tactic = f"Anchor at ({optimal_x:.1f}, {optimal_y:.1f}). Edge (xG diff {team_data['xG_Differential']:.2f})—lock down the danger zone."
        else:
            if x > optimal_x + 10:
                tactic = f"Drop to ({optimal_x:.1f}, {optimal_y:.1f}). Tight (xG diff {team_data['xG_Differential']:.2f})—stay compact."
            elif abs(y - optimal_y) > 10:
                tactic = f"Adjust to ({optimal_x:.1f}, {optimal_y:.1f}). Close (xG diff {team_data['xG_Differential']:.2f})—mark wingers."
            elif x < 20:
                tactic = f"Stay deep at ({x:.1f}, {y:.1f}). Even (xG diff {team_data['xG_Differential']:.2f})—protect the box."
            elif random.random() < 0.3:
                tactic = f"Hold position at ({x:.1f}, {y:.1f}). Tight (xG diff {team_data['xG_Differential']:.2f})—watch for runners."
            else:
                tactic = f"Guard ({optimal_x:.1f}, {optimal_y:.1f}). Close game (xG diff {team_data['xG_Differential']:.2f})—block shots."

    elif role == 'MID':
        if team_data['xG_Differential'] > 0.5:
            if x < 40:
                tactic = f"Push to ({optimal_x:.1f}, {optimal_y:.1f}). Lead (xG diff {team_data['xG_Differential']:.2f})—drive play forward."
            elif abs(y - 34) > 20:
                tactic = f"Move to ({optimal_x:.1f}, {optimal_y:.1f}). Edge (xG diff {team_data['xG_Differential']:.2f})—exploit wide spaces."
            elif x > 70:
                tactic = f"Support attack at ({optimal_x:.1f}, {optimal_y:.1f}). Lead (xG diff {team_data['xG_Differential']:.2f})—feed forwards."
            elif random.random() < 0.3:
                tactic = f"Drop to ({x:.1f}, {y:.1f}) to recycle. Advantage (xG diff {team_data['xG_Differential']:.2f})—keep possession."
            else:
                tactic = f"Control ({optimal_x:.1f}, {optimal_y:.1f}). Dominance (xG diff {team_data['xG_Differential']:.2f})—stretch their midfield."
        else:
            if x < 40:
                tactic = f"Advance to ({optimal_x:.1f}, {optimal_y:.1f}). Tight (xG diff {team_data['xG_Differential']:.2f})—link defense and attack."
            elif abs(y - 34) > 20:
                tactic = f"Shift to ({optimal_x:.1f}, {optimal_y:.1f}). Close (xG diff {team_data['xG_Differential']:.2f})—cover flanks."
            elif x > 70:
                tactic = f"Hold at ({optimal_x:.1f}, {optimal_y:.1f}). Even (xG diff {team_data['xG_Differential']:.2f})—support counters."
            elif random.random() < 0.3:
                tactic = f"Stay at ({x:.1f}, {y:.1f}) to shield. Tight (xG diff {team_data['xG_Differential']:.2f})—break their press."
            else:
                tactic = f"Pivot at ({optimal_x:.1f}, {optimal_y:.1f}). Close game (xG diff {team_data['xG_Differential']:.2f})—maintain balance."

    elif role == 'GK':
        if team_data['xG_Differential'] > 0.5:
            if x > 10 and team == 'Argentina':
                tactic = f"Move to ({optimal_x:.1f}, {optimal_y:.1f}). Lead (xG diff {team_data['xG_Differential']:.2f})—play out confidently."
            elif x < 95 and team == 'France':
                tactic = f"Shift to ({optimal_x:.1f}, {optimal_y:.1f}). Edge (xG diff {team_data['xG_Differential']:.2f})—start attacks."
            elif abs(y - 34) > 5:
                tactic = f"Adjust to ({optimal_x:.1f}, {optimal_y:.1f}). Lead (xG diff {team_data['xG_Differential']:.2f})—cover angles."
            elif random.random() < 0.3:
                tactic = f"Stay at ({x:.1f}, {y:.1f}) to organize. Advantage (xG diff {team_data['xG_Differential']:.2f})—direct defense."
            else:
                tactic = f"Command ({optimal_x:.1f}, {optimal_y:.1f}). Dominance (xG diff {team_data['xG_Differential']:.2f})—distribute accurately."
        else:
            if x > 10 and team == 'Argentina':
                tactic = f"Drop to ({optimal_x:.1f}, {optimal_y:.1f}). Tight (xG diff {team_data['xG_Differential']:.2f})—stay alert."
            elif x < 95 and team == 'France':
                tactic = f"Move to ({optimal_x:.1f}, {optimal_y:.1f}). Close (xG diff {team_data['xG_Differential']:.2f})—anticipate shots."
            elif abs(y - 34) > 5:
                tactic = f"Shift to ({optimal_x:.1f}, {optimal_y:.1f}). Even (xG diff {team_data['xG_Differential']:.2f})—watch crosses."
            elif random.random() < 0.3:
                tactic = f"Hold at ({x:.1f}, {y:.1f}) to organize. Tight (xG diff {team_data['xG_Differential']:.2f})—keep defense tight."
            else:
                tactic = f"Guard ({optimal_x:.1f}, {optimal_y:.1f}). Close game (xG diff {team_data['xG_Differential']:.2f})—make key saves."

    # Player-specific advice based on stats
    if player_data.get('goals', 0) > 2:
        player_advice = "You’re in top scoring form—unleash more shots and test their keeper."
    elif player_data.get('assists', 0) > 1:
        player_advice = "Your vision is key—seek out runners and deliver killer passes."
    elif player_data.get('passes', 0) > 100:
        player_advice = "Master of possession—keep the ball moving and control the game’s rhythm."
    elif player_data.get('avg_rating', 0) > 7.5:
        player_advice = "You’re a standout—step up, inspire the team, and drive us forward."
    else:
        player_advice = "Stay composed and disciplined—focus on teamwork to turn the tide."

    # Combine tactic and advice
    suggestion = f"{tactic} {player_advice}"

    return suggestion, (optimal_x, optimal_y)

def create_pitch_figure(selected_player=None):
    """Create the pitch visualization with player positions and football field markings."""
    traces = []
    
    # Add football field markings on the green background
    # Pitch outline
    traces.append(go.Scatter(x=[0, 105, 105, 0, 0], y=[0, 0, 68, 68, 0], mode="lines",
                             line=dict(color="white", width=2), showlegend=False))
    # Halfway line
    traces.append(go.Scatter(x=[52.5, 52.5], y=[0, 68], mode="lines",
                             line=dict(color="white", width=2, dash="dash"), showlegend=False))
    # Center circle
    theta = np.linspace(0, 2*np.pi, 100)
    center_circle_x = 52.5 + 9.15 * np.cos(theta)
    center_circle_y = 34 + 9.15 * np.sin(theta)
    traces.append(go.Scatter(x=center_circle_x, y=center_circle_y, mode="lines",
                             line=dict(color="white", width=2), showlegend=False))
    # Penalty areas
    traces.append(go.Scatter(x=[0, 16.5, 16.5, 0, 0], y=[13.84, 13.84, 54.16, 54.16, 13.84], mode="lines",
                             line=dict(color="white", width=2), showlegend=False))  # Left
    traces.append(go.Scatter(x=[88.5, 105, 105, 88.5, 88.5], y=[13.84, 13.84, 54.16, 54.16, 13.84], mode="lines",
                             line=dict(color="white", width=2), showlegend=False))  # Right
    # Goal areas
    traces.append(go.Scatter(x=[0, 5.5, 5.5, 0, 0], y=[24.84, 24.84, 43.16, 43.16, 24.84], mode="lines",
                             line=dict(color="white", width=2), showlegend=False))  # Left
    traces.append(go.Scatter(x=[99.5, 105, 105, 99.5, 99.5], y=[24.84, 24.84, 43.16, 43.16, 24.84], mode="lines",
                             line=dict(color="white", width=2), showlegend=False))  # Right
    # Goals
    traces.append(go.Scatter(x=[0, 0], y=[30.34, 37.66], mode="lines",
                             line=dict(color="white", width=4), showlegend=False))  # Left
    traces.append(go.Scatter(x=[105, 105], y=[30.34, 37.66], mode="lines",
                             line=dict(color="white", width=4), showlegend=False))  # Right
    
    # Player positions
    for player, (x, y) in player_positions.items():
        color = "blue" if player in argentina_players else "red"
        traces.append(go.Scatter(
            x=[x], y=[y], mode="markers+text", text=[player], textposition="top center",
            marker=dict(size=12, color=color, line=dict(width=2, color='black')),
            customdata=[player], hovertemplate=f"<b>{player}</b><br>x: %{{x:.2f}}, y: %{{y:.2f}}<extra></extra>"
        ))
        
        if selected_player == player and player in argentina_players:
            team = 'Argentina'
            _, (opt_x, opt_y) = ai_tactic_and_position(player, team)
            traces.append(go.Scatter(
                x=[x, opt_x], y=[y, opt_y], mode="lines+markers",
                line=dict(color="yellow", width=2, dash="dash"),
                marker=dict(size=8, color="yellow")
            ))
    
    layout = go.Layout(
        xaxis=dict(range=[0, FIELD_LENGTH], showgrid=False, zeroline=True, visible=False),
        yaxis=dict(range=[0, FIELD_WIDTH], showgrid=False, zeroline=True, visible=False),
        plot_bgcolor="green", height=500, margin=dict(l=20, r=20, t=20, b=20),
        title="."
    )
    return go.Figure(data=traces, layout=layout)

# -------------------------
# Dash Application
# -------------------------
app = dash.Dash(__name__)
app.title = "FIFA World Cup Final 2022 Simulation"

app.layout = html.Div([
    html.H1("Real-Time FIFA World Cup Final 2022 Simulation"),
    dcc.Interval(id='interval-component', interval=2000, n_intervals=0),  # Update every 2 seconds
    html.Div([
        html.Div([dcc.Graph(id='pitch-graph')],
                 style={'width': '75%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.H3("Player Details"),
            html.Div(id='player-details', children="Click a player marker to see stats.",
                     style={'border': '2px solid #333', 'padding': '15px', 'backgroundColor': '#f9f9f9',
                            'borderRadius': '10px', 'boxShadow': '2px 2px 8px rgba(0,0,0,0.1)'})
        ], style={'width': '23%', 'display': 'inline-block', 'marginLeft': '2%', 'verticalAlign': 'top'})
    ]),
    html.Div([
        html.Span("Argentina", style={'color': 'blue', 'fontSize': '20px', 'position': 'absolute', 'left': '10%', 'bottom': '5px'}),
        html.Span("France", style={'color': 'red', 'fontSize': '20px', 'position': 'absolute', 'right': '45%', 'bottom': '5px'})
    ], style={'position': 'relative', 'height': '40px'})
], style={'padding': '20px'})

# -------------------------
# Callbacks
# -------------------------
@app.callback(
    Output('pitch-graph', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('pitch-graph', 'clickData')]
)
def update_pitch_graph(n, clickData):
    update_player_positions(n)
    selected_player = clickData['points'][0]['customdata'] if clickData else None
    return create_pitch_figure(selected_player)

@app.callback(
    Output('player-details', 'children'),
    Input('pitch-graph', 'clickData')
)
def display_player_details(clickData):
    if not clickData:
        return "Click a player marker to see stats."
    
    player = clickData['points'][0]['customdata']
    stats = player_past_stats.get(player, {})
    x, y = player_positions[player]
    team = 'Argentina' if player in argentina_players else 'France'
    tactic, (opt_x, opt_y) = ai_tactic_and_position(player, team)
    
    details = [
        html.H4(player, style={'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P(f"Current Position: x={x:.1f}, y={y:.1f}", style={'fontSize': '14px', 'color': '#555'}),
        html.P(f"Matches: {stats.get('matches', 0)}", style={'fontSize': '14px', 'color': '#555'}),
        html.P(f"Goals: {stats.get('goals', 0)}", style={'fontSize': '14px', 'color': '#555'}),
        html.P(f"Assists: {stats.get('assists', 0)}", style={'fontSize': '14px', 'color': '#555'}),
        html.P(f"Passes: {stats.get('passes', 0)}", style={'fontSize': '14px', 'color': '#555'}),
        html.P(f"Avg Rating: {stats.get('avg_rating', 0.0)}", style={'fontSize': '14px', 'color': '#555'}),
        html.Div([
            html.Strong("Tactical Plan: ", style={'color': '#e74c3c'}),
            html.Span(tactic, style={'backgroundColor': '#e74c3c', 'color': 'white', 'padding': '5px 10px',
                                     'borderRadius': '5px', 'display': 'inline-block'})
        ], style={'marginTop': '15px', 'fontSize': '14px'}),
        html.P(f"Optimal Position: x={opt_x:.1f}, y={opt_y:.1f}", style={'fontSize': '14px', 'color': '#555', 'marginTop': '10px'})
    ]
    return details

# -------------------------
# Run the Application
# -------------------------
if __name__ == '__main__':
    app.run_server(debug=True)