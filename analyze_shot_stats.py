"""
Shot Data Statistical Analysis

Features:
- Analyze shot distribution across competitions (e.g., La Liga, World Cup)
- Analyze player statistics
- Calculate various metrics
- Generate visualizations

Usage: python src/analyze_shot_stats.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Add project path
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    COMPETITIONS,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PITCH_LENGTH,
    PITCH_WIDTH
)

# Set font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_all_shot_data():
    """Load all extracted shot data"""
    print("Loading shot data...")

    all_shots = []
    total_files = 0

    # Traverse all data in data/raw
    for season_dir in Path(RAW_DATA_DIR).iterdir():
        if season_dir.is_dir():
            for csv_file in season_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    if len(df) > 0:
                        all_shots.append(df)
                        total_files += 1
                        print(f"  Loaded: {csv_file.name} ({len(df)} shot records)")
                except Exception as e:
                    print(f"  Error: Cannot load {csv_file.name}: {e}")

    if not all_shots:
        print("Error: No shot data files found")
        return pd.DataFrame()

    # Combine all data
    combined_df = pd.concat(all_shots, ignore_index=True)
    print(f"\n[SUCCESS] Total loaded {len(combined_df)} shot records from {total_files} files")

    return combined_df

def basic_statistics(df):
    """Basic statistical analysis"""
    print("\n" + "="*60)
    print("Basic Statistics")
    print("="*60)

    # Basic info
    print(f"Total shots: {len(df)}")
    print(f"Total goals: {df['goal'].sum()} ({df['goal'].mean()*100:.1f}%)")

    # Competition statistics
    competition_stats = df.groupby(['competition_name', 'season_name']).agg({
        'id': 'count',
        'goal': ['sum', 'mean']
    }).round(2)

    competition_stats.columns = ['SHOTS_COUNT', 'GOALS_COUNT', 'GOAL_RATE']
    competition_stats['GOAL_RATE_PCT'] = competition_stats['GOAL_RATE'] * 100
    competition_stats = competition_stats[['SHOTS_COUNT', 'GOALS_COUNT', 'GOAL_RATE_PCT']]
    competition_stats = competition_stats.sort_values('SHOTS_COUNT', ascending=False)

    print("\nSeason-League Shot Statistics:")
    print(competition_stats)

    # Player statistics
    player_stats = df.groupby(['player_id', 'player', 'team']).agg({
        'id': 'count',
        'goal': 'sum'
    }).round(2)

    player_stats.columns = ['SHOTS_COUNT', 'GOALS_COUNT']
    player_stats['GOAL_RATE_PCT'] = (player_stats['GOALS_COUNT'] / player_stats['SHOTS_COUNT'] * 100).round(2)

    # Top 10 players with most shots
    player_stats_reset = player_stats.reset_index()
    top_scorers = player_stats_reset.sort_values('SHOTS_COUNT', ascending=False).head(10)
    print("\nMost Shots Players (Top 10):")
    try:
        for _, row in top_scorers.iterrows():
            print("  {} ({}): {} shots, {} goals, {:.1f}%".format(
            row['player'], row['team'], row['SHOTS_COUNT'], row['GOALS_COUNT'], row['GOAL_RATE_PCT']))
    except UnicodeEncodeError:
        print("  [Data contains non-ASCII characters]")

    # Most accurate players (at least 10 shots)
    top_shooters = player_stats_reset[player_stats_reset['SHOTS_COUNT'] >= 10].sort_values('GOAL_RATE_PCT', ascending=False).head(10)
    print("\nMost Accurate Players (Top 10, at least 10 shots):")
    try:
        for _, row in top_shooters.iterrows():
            print("  {} ({}): {} shots, {} goals, {:.1f}%".format(
            row['player'], row['team'], row['SHOTS_COUNT'], row['GOALS_COUNT'], row['GOAL_RATE_PCT']))
    except UnicodeEncodeError:
        print("  [Data contains non-ASCII characters]")

    return competition_stats, player_stats

def competition_distribution(df):
    """Analyze competition distribution"""
    print("\n" + "="*60)
    print("Competition Distribution Analysis")
    print("="*60)

    # Statistics by league
    league_stats = df.groupby('competition_name').agg({
        'id': 'count',
        'goal': ['sum', 'mean']
    }).round(2)

    league_stats.columns = ['SHOTS_COUNT', 'GOALS_COUNT', 'GOAL_RATE']
    league_stats = league_stats.sort_values('SHOTS_COUNT', ascending=False)

    print("\nLeague Shot Distribution:")
    for league, row in league_stats.iterrows():
        print(f"  {league}: {row['SHOTS_COUNT']} shots, {row['GOALS_COUNT']} goals, {row['GOAL_RATE']*100:.1f}%")

    # Create charts
    plt.figure(figsize=(12, 6))

    # Shots count bar chart
    plt.subplot(1, 2, 1)
    league_shots = league_stats['SHOTS_COUNT'].sort_values(ascending=False)
    bars = plt.bar(range(len(league_shots)), league_shots, color='skyblue')
    plt.title('Shots Distribution by League', fontsize=14, fontweight='bold')
    plt.xlabel('League')
    plt.ylabel('Shots Count')
    plt.xticks(range(len(league_shots)), league_shots.index, rotation=45)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')

    # Goal rate comparison
    plt.subplot(1, 2, 2)
    league_goal_rates = league_stats['GOAL_RATE'].sort_values(ascending=False) * 100
    bars = plt.bar(range(len(league_goal_rates)), league_goal_rates, color='lightcoral')
    plt.title('Goal Rate by League', fontsize=14, fontweight='bold')
    plt.xlabel('League')
    plt.ylabel('Goal Rate (%)')
    plt.xticks(range(len(league_goal_rates)), league_goal_rates.index, rotation=45)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"{PROCESSED_DATA_DIR}/competition_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nChart saved to: {PROCESSED_DATA_DIR}/competition_distribution.png")

    return league_stats

def player_analysis(df):
    """Player analysis"""
    print("\n" + "="*60)
    print("Player Analysis")
    print("="*60)

    # Count players with shots per team
    player_per_team = df.groupby(['team', 'player_id']).size().groupby('team').count()
    team_player_stats = pd.DataFrame({
        'TEAM': player_per_team.index,
        'PLAYERS_WITH_SHOTS_COUNT': player_per_team.values
    })

    # Sort by number of players with shots
    team_player_stats = team_player_stats.sort_values('PLAYERS_WITH_SHOTS_COUNT', ascending=False)

    print("\nTeams with Most Players Taking Shots (Top 10):")
    for _, row in team_player_stats.head(10).iterrows():
        print(f"  {row['TEAM']}: {row['PLAYERS_WITH_SHOTS_COUNT']} players")

    # Player activity analysis
    total_players_with_shots = df['player_id'].nunique()
    total_teams = df['team'].nunique()
    avg_players_per_team = total_players_with_shots / total_teams

    print(f"\nPlayer Activity:")
    print(f"  Total players with shots: {total_players_with_shots}")
    print(f"  Total teams with shots: {total_teams}")
    print(f"  Average players per team with shots: {avg_players_per_team:.1f}")

    # Create charts
    plt.figure(figsize=(12, 6))

    # Teams with most players taking shots (Top 10)
    plt.subplot(1, 2, 1)
    top_teams = team_player_stats.head(10)
    bars = plt.barh(range(len(top_teams)), top_teams['PLAYERS_WITH_SHOTS_COUNT'], color='lightgreen')
    plt.title('Teams with Most Players Taking Shots (Top 10)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Players')
    plt.yticks(range(len(top_teams)), top_teams['TEAM'])

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2.,
                f'{int(width)}', ha='left', va='center')

    # Player shots distribution
    plt.subplot(1, 2, 2)
    shots_per_player = df['player_id'].value_counts()
    plt.hist(shots_per_player, bins=20, color='orange', edgecolor='black', alpha=0.7)
    plt.title('Player Shots Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Shots Count')
    plt.ylabel('Number of Players')
    plt.axvline(shots_per_player.mean(), color='red', linestyle='--',
               label=f'Average: {shots_per_player.mean():.1f} shots')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{PROCESSED_DATA_DIR}/player_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nChart saved to: {PROCESSED_DATA_DIR}/player_analysis.png")

    return {
        'total_players_with_shots': total_players_with_shots,
        'total_teams': total_teams,
        'avg_players_per_team': avg_players_per_team,
        'team_player_stats': team_player_stats
    }

def location_analysis(df):
    """Shot location analysis (if data exists)"""
    print("\n" + "="*60)
    print("Shot Location Analysis")
    print("="*60)

    # Check if location data exists
    if 'location_x' in df.columns and 'location_y' in df.columns:
        valid_shots = df.dropna(subset=['location_x', 'location_y'])

        if len(valid_shots) > 0:
            # Calculate distance to goal for each shot
            valid_shots['distance_from_goal'] = np.sqrt(
                (valid_shots['location_x'] - 120)**2 +
                (valid_shots['location_y'] - 40)**2
            )

            print("\nShot Location Statistics:")
            print(f"  Closest shot distance: {valid_shots['distance_from_goal'].min():.1f} meters")
            print(f"  Farthest shot distance: {valid_shots['distance_from_goal'].max():.1f} meters")
            print(f"  Average shot distance: {valid_shots['distance_from_goal'].mean():.1f} meters")

            # Create shot heatmap
            plt.figure(figsize=(12, 8))

            # Create pitch background
            pitch = plt.Rectangle((0, 0), PITCH_LENGTH, PITCH_WIDTH,
                                fill=True, facecolor='green', alpha=0.3)
            plt.gca().add_patch(pitch)

            # Mark goal
            plt.plot([120, 120], [36, 44], 'k-', linewidth=3)  # Goal line
            plt.plot([118, 122], [36, 36], 'k-', linewidth=2)  # Goal post
            plt.plot([118, 122], [44, 44], 'k-', linewidth=2)  # Goal post

            # Plot shot locations
            goals = valid_shots[valid_shots['goal'] == 1]
            shots = valid_shots[valid_shots['goal'] == 0]

            plt.scatter(shots['location_x'], shots['location_y'],
                       c='blue', alpha=0.5, s=10, label='Missed shots')
            plt.scatter(goals['location_x'], goals['location_y'],
                       c='red', alpha=0.8, s=20, label='Goals')

            plt.title('Shot Heatmap', fontsize=14, fontweight='bold')
            plt.xlabel('X coordinate (meters)')
            plt.ylabel('Y coordinate (meters)')
            plt.xlim(-5, PITCH_LENGTH + 5)
            plt.ylim(-5, PITCH_WIDTH + 5)
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.savefig(f"{PROCESSED_DATA_DIR}/shot_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()

            print(f"\nShot heatmap saved to: {PROCESSED_DATA_DIR}/shot_heatmap.png")

            return valid_shots
        else:
            print("  Warning: No valid shot location data")
            return None
    else:
        print("  Warning: Missing location information in data")
        return None

def save_summary_report(competition_stats, player_stats, player_analysis_result, location_data=None):
    """Save statistical report"""
    print("\n" + "="*60)
    print("Saving Statistical Report")
    print("="*60)

    # Create processed directory if not exists
    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)

    # Generate report data
    competition_stats_flat = competition_stats.reset_index()
    player_stats_flat = player_stats.reset_index()

    report_data = {
        'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_shots': int(competition_stats['SHOTS_COUNT'].sum()),
        'total_goals': int(competition_stats['GOALS_COUNT'].sum()),
        'competition_stats': competition_stats_flat.to_dict('records'),
        'player_analysis': {
            'total_players_with_shots': int(player_analysis_result['total_players_with_shots']),
            'total_teams': int(player_analysis_result['total_teams']),
            'avg_players_per_team': float(player_analysis_result['avg_players_per_team'])
        },
        'top_scorers': player_stats_flat.sort_values('SHOTS_COUNT', ascending=False).head(10).to_dict('records'),
        'top_shooters': player_stats_flat[player_stats_flat['SHOTS_COUNT'] >= 10]
                                  .sort_values('GOAL_RATE_PCT', ascending=False)
                                  .head(10)
                                  .to_dict('records')
    }

    if location_data is not None:
        report_data['location_stats'] = {
            'min_distance': location_data['distance_from_goal'].min(),
            'max_distance': location_data['distance_from_goal'].max(),
            'avg_distance': location_data['distance_from_goal'].mean()
        }

    # Save JSON report
    report_file = Path(PROCESSED_DATA_DIR) / 'shot_statistics_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"Statistical report saved to: {report_file}")

    # Generate CSV files
    # League statistics
    league_report = competition_stats.reset_index()
    league_report.to_csv(Path(PROCESSED_DATA_DIR) / 'league_statistics.csv', index=False, encoding='utf-8')

    # Player statistics Top 50
    player_report = player_stats.reset_index().head(50)
    player_report.to_csv(Path(PROCESSED_DATA_DIR) / 'player_statistics_top50.csv', index=False, encoding='utf-8')

    print(f"League statistics CSV saved to: {PROCESSED_DATA_DIR}/league_statistics.csv")
    print(f"Player statistics CSV saved to: {PROCESSED_DATA_DIR}/player_statistics_top50.csv")

def main():
    """Main function"""
    print("\n" + "="*80)
    print("*" + " " * 30 + "Shot Data Statistical Analysis Tool")
    print("="*80)

    # Check processed directory
    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_all_shot_data()

    if len(df) == 0:
        print("\n[ERROR] No shot data found. Please run extract_shots.py first")
        return

    # Perform various analyses
    competition_stats, player_stats = basic_statistics(df)
    league_stats = competition_distribution(df)
    player_analysis_result = player_analysis(df)
    location_data = location_analysis(df)

    # Save report
    save_summary_report(competition_stats, player_stats, player_analysis_result, location_data)

    print("\n" + "="*80)
    print("[SUCCESS] All analysis completed!")
    print("\nGenerated files:")
    print(f"  - Competition distribution chart: {PROCESSED_DATA_DIR}/competition_distribution.png")
    print(f"  - Player analysis chart: {PROCESSED_DATA_DIR}/player_analysis.png")
    print(f"  - Shot heatmap: {PROCESSED_DATA_DIR}/shot_heatmap.png")
    print(f"  - Statistical report: {PROCESSED_DATA_DIR}/shot_statistics_report.json")
    print(f"  - League statistics: {PROCESSED_DATA_DIR}/league_statistics.csv")
    print(f"  - Player statistics: {PROCESSED_DATA_DIR}/player_statistics_top50.csv")
    print("="*80)


if __name__ == "__main__":
    main()