SELECT

/***************** match index info ************/

matches.match_id                               as MatchID,
matches.start_time                             as Start,
matches.duration                               as Duration,
match_patch.patch                              as patch,


/***************** match results ***************/

radiant_win                                    as Win,
avg(kills)                                     as Kills,
avg(deaths)                                    as Deaths,
avg(assists)                                   as Assists,

/***************** match details ***************/

avg(stuns)                                     as Stuns,
avg(denies)                                    as Denies,
avg(camps_stacked)                             as Stacks,
avg(gold_per_min)                              as GPM,
avg(hero_damage)                               as HeroDamage,
avg(hero_healing)                              as HeroHealing,
avg(rune_pickups)                              as Runes,
avg(array_length(buyback_log, 1))              as BuyBackS,
avg(array_length(obs_log, 1))                  as WardPlaced,
avg((killed->>'npc_dota_observer_wards')::int) as WardDestroyed,

count(distinct matches.match_id) count,
stddev(kills::numeric) stddev


FROM matches
JOIN match_patch using(match_id)
JOIN leagues using(leagueid)
JOIN player_matches using(match_id)
JOIN heroes on heroes.id = player_matches.hero_id
LEFT JOIN notable_players ON notable_players.account_id = player_matches.account_id AND notable_players.locked_until = (SELECT MAX(locked_until) FROM notable_players)
LEFT JOIN teams using(team_id)
WHERE TRUE
AND deaths IS NOT NULL
AND match_patch.patch >= '7.10'
AND match_patch.patch <= '7.10'
AND (player_matches.player_slot < 128) = true
AND leagues.tier = 'professional'
GROUP BY match_patch.patch, matches.match_id
HAVING count(distinct matches.match_id) >= 1

LIMIT 10000
