
import pandas as pd
li = 'nsmc-sampled','nsmc-rtt','slack-rtt','slack-sampled','petition-sampled','petition-rtt'

df = pd.read_csv('./data/train.csv')
nsmc_rtt = df[df['source'].str.contains('nsmc-rtt')]
nsmc_sampled = df[df['source'].str.contains('nsmc-sampled')]

slack_rtt = df[df['source'].str.contains('slack-rtt')]
slack_sampled = df[df['source'].str.contains('slack-sampled')]

petition_sampled = df[df['source'].str.contains('petition-sampled')]
petition_rtt = df[df['source'].str.contains('petition-rtt')]

nsmc_rtt.to_csv('./data/nsmc_rtt.csv')
nsmc_sampled.to_csv('./data/nsmc_sampled.csv')
slack_rtt.to_csv('./data/slack_rtt.csv')
slack_sampled.to_csv('./data/slack_sampled.csv')
petition_sampled.to_csv('./data/petition_sampled.csv')
petition_rtt.to_csv('./data/petition_rtt.csv')

df = pd.read_csv('./data/dev.csv')
nsmc_rtt = df[df['source'].str.contains('nsmc-rtt')]
nsmc_sampled = df[df['source'].str.contains('nsmc-sampled')]

slack_rtt = df[df['source'].str.contains('slack-rtt')]
slack_sampled = df[df['source'].str.contains('slack-sampled')]

petition_sampled = df[df['source'].str.contains('petition-sampled')]
petition_rtt = df[df['source'].str.contains('petition-rtt')]

nsmc_rtt.to_csv('./data/nsmc_rtt_dev.csv')
nsmc_sampled.to_csv('./data/nsmc_sampled_dev.csv')
slack_rtt.to_csv('./data/slack_rtt_dev.csv')
slack_sampled.to_csv('./data/slack_sampled_dev.csv')
petition_sampled.to_csv('./data/petition_sampled_dev.csv')
petition_rtt.to_csv('./data/petition_rtt_dev.csv')



