import os
import warnings
import json
import numpy as np
import pandas as pd
#from pandas.core.common import SettingWithCopyWarning
from tqdm import tqdm
from collections import defaultdict
# import socceraction.spadl as spadl
#warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def load_wyscout_dataset(dataset_name='Italy'):
    # loading the competitions data
    competitions={}
    with open('./data/competitions.json') as json_data:
        competitions = json.load(json_data)
    competition_dict = defaultdict(dict)
    for competition in competitions:
        competition_id = competition['wyId']
        competition_dict[competition_id] = competition
    
    # loading the teams data
    teams={}
    with open('./data/teams.json') as json_data:
        teams = json.load(json_data)
    team_dict = defaultdict(dict)
    for team in teams:
        team_id = team['wyId']
        team_dict[team_id] = team
                                   
    # loading the players data
    with open('./data/players.json') as json_data:
        players = json.load(json_data)
    
    player_dict = defaultdict(dict)
    for player in players:
        player_id = player['wyId']
        player_dict[player_id] = player

    # loading the matches data
    matches = {}
    with open('./data/matches/matches_%s.json' %dataset_name) as json_data:
        matches = json.load(json_data)

    match_dict = defaultdict(dict)                       
    for match in matches:
        match_id = match['wyId']
        match_dict[match_id] = match

    # loading the events data
    events = {}
    with open('./data/events/events_%s.json' %dataset_name) as json_data:
        events = json.load(json_data)
    
    match_event_dict = defaultdict(list)
    for event in events:
        match_id = event['matchId']
        match_event_dict[match_id].append(event)
    
    return competition_dict, team_dict, player_dict, match_dict, match_event_dict


def refine_and_save_events(dataset_name='World_Cup'):
    competition_dict, team_dict, player_dict, match_dict, match_event_dict = load_wyscout_dataset(dataset_name)
    if not os.path.exists('data/refined_events'):
        os.mkdir('data/refined_events')
    if not os.path.exists(f'data/refined_events/{dataset_name}'):
        os.mkdir(f'data/refined_events/{dataset_name}')

    match_df = refine_matches(team_dict, match_dict)
    match_df.to_csv(f'data/refined_events/{dataset_name}/matches.csv', encoding='utf-8-sig')
    
    dataset_names = ['Italy', 'England', 'Spain', 'France', 'Germany', 'European_Championship', 'World_Cup']
    competition_names = [competition['name'] for competition in competition_dict.values()]
    competition_dict = dict(zip(dataset_names, competition_names))

    for match_id in tqdm(match_dict.keys(), desc=f"{competition_dict[dataset_name]:23s}"):
        events = refine_match_events(team_dict, player_dict, match_dict, match_event_dict, match_id)
        events.to_pickle(f'data/refined_events/{dataset_name}/{match_id}.pkl')


def refine_matches(team_dict, match_dict):
    cols = [
        'gameweek', 'datetime', 'venue',
        'team1_id', 'team1_name', 'team1_goals',
        'team2_id', 'team2_name', 'team2_goals', 'duration'
    ]
    match_df = pd.DataFrame(columns=cols)

    for match_id in match_dict.keys():
        if match_id == 0:
            continue

        match = match_dict[match_id]
        
        team_names, score = match['label'].split(', ')
        team1_name, team2_name = team_names.encode('ascii', 'strict').decode('unicode-escape').split(' - ')
        team1_goals, team2_goals = score.split(' - ')
        team1_id, team2_id = [int(i) for i in match['teamsData'].keys()]
        if team_dict[team1_id]['name'] != team1_name:
            team1_id, team2_id = team2_id, team1_id

        duration = match['duration']
        if duration != 'Regular':
            team2_goals = team2_goals[:-4]

        match_df.loc[match_id] = [
            match['gameweek'], match['dateutc'], 
            match['venue'].encode('ascii', 'strict').decode('unicode-escape'),
            team1_id, team1_name, int(team1_goals),
            team2_id, team2_name, int(team2_goals), duration
        ]
    
    match_df.index.name = 'match_id'
    return match_df.sort_values('datetime')


def refine_match_events(team_dict, player_dict, match_dict, match_event_dict, match_id=2576335):
    tag_dict = pd.read_csv('data/tags2name.csv', index_col=0, header=0)['Description'].to_dict()

    col_dict = {
        'id': 'event_id', 'teamId': 'team_id', 'playerId': 'player_id',
        'eventId': 'type_id', 'subEventId': 'sub_type_id',
        'matchId': 'match_id', 'matchPeriod': 'period', 'eventSec': 'time',
        'eventName': 'event_type', 'subEventName': 'sub_event_type', 'tags': 'tags'
    }
    events = pd.DataFrame(match_event_dict[match_id]).rename(columns=col_dict)
    events['time'] = events['time'].round(3)

    events['team_name'] = events['team_id'].apply(
        lambda x: team_dict[x]['name'].encode('ascii', 'strict').decode('unicode-escape')
    )
    events['player_name'] = events['player_id'].apply(
        lambda x: player_dict[x]['shortName'].encode('ascii', 'strict').decode('unicode-escape')
        if x != 0 else np.nan
    )
    events.replace({'Free Kick': 'Free kick'}, inplace=True)
    events['tags'] = events['tags'].apply(lambda x: [tag_dict[tag['id']] for tag in x])

    for tags in events[events['event_type'] == 'Save attempt']['tags']:
        if 'Goal' in tags:
            tags.remove('Goal')

    events['start_x'] = [float(x[0]['x']) for x in events['positions']]
    events['start_y'] = [float(x[0]['y']) for x in events['positions']]
    events['end_x'] = [float(x[1]['x']) if len(x) > 1 else np.nan for x in events['positions']]
    events['end_y'] = [float(x[1]['y']) if len(x) > 1 else np.nan for x in events['positions']]
    
    events.loc[events['sub_event_type'] == 'Goal kick', ['start_x', 'start_y']] = [0, 50]
    events.loc[events['event_type'].isin(['Save attempt', 'Goalkeeper leaving line']), ['start_x', 'start_y']] = [0, 50]
    events.loc[events['event_type'] == 'Shot', ['end_x', 'end_y']] = [100, 50]
    events.loc[events['sub_event_type'].isin(['Free kick shot', 'Penalty']), ['end_x', 'end_y']] = [100, 50]
    events.loc[events['event_type'].isin(['Foul', 'Offside']), ['end_x', 'end_y']] = np.nan
    events.loc[events['sub_event_type'] == 'Ball out of the field', ['end_x', 'end_y']] = np.nan
    events.loc[events['sub_event_type'] == 'Whistle', ['start_x', 'start_y', 'end_x', 'end_y']] = np.nan

    events[['start_x', 'end_x']] = events[['start_x', 'end_x']] * 1.04
    events[['start_y', 'end_y']] = (100 - events[['start_y', 'end_y']]) * 0.68

    team2_name = events['team_name'].unique()[1]
    team2_x = events.loc[events['team_name'] == team2_name, ['start_x', 'end_x']]
    team2_y = events.loc[events['team_name'] == team2_name, ['start_y', 'end_y']]
    events.loc[events['team_name'] == team2_name, ['start_x', 'end_x']] = 104 - team2_x
    events.loc[events['team_name'] == team2_name, ['start_y', 'end_y']] = 68 - team2_y

    gk_actions = events[events['event_type'].isin(['Save attempt', 'Goalkeeper leaving line'])]
    for i, event in gk_actions.iterrows():
        if i == events.index[-1] or events.at[i, 'period'] != events.at[i+1, 'period']:
            events.at[i, 'end_x'] = np.nan
            events.at[i, 'end_y'] = np.nan
        elif events.at[i+1, 'event_type'] != 'Free kick':
            if event['event_type'] == 'Goalkeeper leaving line' or 'Accurate' in event['tags']:
                events.at[i, 'end_x'] = events.at[i+1, 'start_x']
                events.at[i, 'end_y'] = events.at[i+1, 'start_y']
        else:
            events.at[i, 'end_x'] = np.nan
            events.at[i, 'end_y'] = np.nan

    shots = events[(events['event_type'] == 'Shot') | (events['sub_event_type'].isin(['Free kick', 'Penalty']))]
    for i, event in shots.iterrows():
        if i < events.index[-1] and events.at[i, 'period'] == events.at[i, 'period']:
            if 'Blocked' in event['tags']:
                events.at[i, 'end_x'] = events.at[i+1, 'start_x']
                events.at[i, 'end_y'] = events.at[i+1, 'start_y']
                if i+1 < events.index[-1] and events.at[i+1, 'period'] == events.at[i+2, 'period']:
                    events.at[i+1, 'end_x'] = events.at[i+2, 'start_x']
                    events.at[i+1, 'end_y'] = events.at[i+2, 'start_y']
                else:
                    events.at[i+1, 'end_x'] = np.nan
                    events.at[i+1, 'end_y'] = np.nan
            elif events.at[i+1, 'sub_event_type'] == 'Touch':
                events.at[i+1, 'end_x'] = np.nan
                events.at[i+1, 'end_y'] = np.nan

    for period in events['period'].unique():
        period_events = events[events['period'] == period]
        to_corners = period_events[(period_events['end_x'].isin([0, 104])) & (period_events['end_y'].isin([0, 68]))]
        for i, event in to_corners.iterrows():
            if i == period_events.index[-1]:
                events.at[i, 'end_x'] = np.nan
                events.at[i, 'end_y'] = np.nan
            elif events.at[i+1, 'sub_event_type'] == 'Corner':
                events.at[i, 'end_x'] = events.at[i+1, 'start_x']
                events.at[i, 'end_y'] = events.at[i, 'start_y']
            else:
                events.at[i, 'end_x'] = events.at[i+1, 'start_x']
                events.at[i, 'end_y'] = events.at[i+1, 'start_y']
    
    team2_x = events.loc[events['team_name'] == team2_name, ['start_x', 'end_x']]
    team2_y = events.loc[events['team_name'] == team2_name, ['start_y', 'end_y']]
    events.loc[events['team_name'] == team2_name, ['start_x', 'end_x']] = 104 - team2_x
    events.loc[events['team_name'] == team2_name, ['start_y', 'end_y']] = 68 - team2_y

    cols = [
        'match_id', 'event_id', 'period', 'time', 'team_id', 'team_name', 'player_id', 'player_name',
        'event_type', 'sub_event_type', 'tags', 'start_x', 'start_y', 'end_x', 'end_y'
    ]
    events = convert_substitution_records(match_dict[match_id], events[cols])
    return events

def convert_substitution_records(match, events):
    match_id = match['wyId']
    subs_list = []

    for team_id in match['teamsData'].keys():
        team_sub_dict = match['teamsData'][team_id]['formation']['substitutions']
        try:
            team_subs = pd.DataFrame(team_sub_dict)
            team_subs['match_id'] = int(match_id)
            team_subs['team_id'] = int(team_id)
            subs_list.append(team_subs)
        except ValueError:
            continue
    
    match_subs = pd.concat(subs_list, ignore_index=True).rename(
        columns={'playerIn': 'in_player_id', 'playerOut': 'out_player_id'})

    match_team_df = events[['team_id', 'team_name']].drop_duplicates()
    match_subs = pd.merge(match_subs, match_team_df)

    match_player_df = events[['player_id', 'player_name']].drop_duplicates()
    match_subs = pd.merge(match_subs, match_player_df, left_on='in_player_id', right_on='player_id')
    match_subs.rename(columns={'player_name': 'in_player_name'}, inplace=True)
    match_subs = pd.merge(match_subs, match_player_df, left_on='out_player_id', right_on='player_id')
    match_subs.rename(columns={'player_name': 'out_player_name'}, inplace=True)
    match_subs['event_id'] = 0

    if match['duration'] == 'Regular':
        bins = [0, 46, 180]
        periods = ['1H', '2H']
    else:
        bins = [0, 46, 91, 106, 180]
        periods = ['1H', '2H', 'E1', 'E2']

    period_starts = dict(zip(periods, bins[:-1]))
    match_subs['period'] = pd.cut(match_subs['minute'], bins, right=False, labels=periods)
    record_list = []

    for _, record in match_subs.iterrows():
        period_events = events[events['period'] == record['period']]

        out_player_last_time = period_events[period_events['player_id'] == record['out_player_id']]['time'].max()
        in_player_first_time = period_events[period_events['player_id'] == record['in_player_id']]['time'].min()
        official_sub_time = (record['minute'] - period_starts[record['period']]) * 60
        record['time'] = float(min(max(official_sub_time, out_player_last_time), in_player_first_time) // 10 * 10)

        sub_in_record = record[events.columns.tolist()[:6] + ['in_player_id', 'in_player_name']].copy()
        sub_in_record.index = events.columns[:8]
        sub_in_record['event_type'] = 'Substitution'
        sub_in_record['sub_event_type'] = 'Player in'
        sub_in_record['tags'] = [record['out_player_id']]
        record_list.append(sub_in_record)

        sub_out_record = record[events.columns.tolist()[:6] + ['out_player_id', 'out_player_name']].copy()
        sub_out_record.index = events.columns[:8]
        sub_out_record['event_type'] = 'Substitution'
        sub_out_record['sub_event_type'] = 'Player out'
        sub_out_record['tags'] = [record['in_player_id']]
        record_list.append(sub_out_record)

    match_subs = pd.concat(record_list, axis=1).T
    return pd.concat([events, match_subs]).sort_values(['period', 'time'], ignore_index=True)
