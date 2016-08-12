user_att = "SELECT user.id, \
          user.created_at, \
         if((2016 -  year(birthday)) > 80, 0, 2016 -  year(birthday)) as year, \
         gender = 'm' as gender, \
         substring_index(substring_index(ref.value,'_',2),'_',-1) as refname, \
         if (ref.type = 'organic', 1, 0) as type, \
         DATEDIFF(paym.created_at, user.created_at) as FP FROM user \
  LEFT JOIN ref ON ref.id = user.ref_id \
  LEFT JOIN (select user_id, created_at FROM payment \
                where user_id IS NOT NULL \
                order by user_id, created_at) as paym ON paym.user_id = user.id \
  WHERE user.created_at > %s and DATE(user.created_at) < DATE_SUB(CURDATE(), INTERVAL 2 DAY) and user.level_id != 2161"

user_lvl = "select T.user_id, T.score, T.moves, TIME_TO_SEC(timediff(T.updated_at, T.created_at)) as gtime, T.name as lvl, \
   T.is_win = 1 as win, \
   T.is_win = 0 and T.loss_reason = '' and T.moves = 0 as drp, \
   T.loss_reason = 'moves' as moves, \
   T.loss_reason = 'bombs' as bombs,  \
   T.loss_reason = 'exit' as ext, \
   T.loss_reason = 'seconds' as secs  \
  from (select ul.user_id, ul.score, ul.moves, ul.is_win, ul.loss_reason, ul.created_at, ul.updated_at, level.name FROM user_level ul \
            left join level ON ul.level_id = level.id \
            left join user ON ul.user_id = user.id \
          where level.name BETWEEN 1 AND 12 AND user.created_at > %s and TIME_TO_SEC(TIMEDIFF(ul.created_at, user.created_at)) < 86400 and DATE(user.created_at) < DATE_SUB(CURDATE(), INTERVAL 2 DAY) and user.level_id != 2161) T "

user_ssn = "select T.user_id,  \
          IF(T.ttime >= 0 AND T.ttime < 10800, T.ping_count, 0) as 0_3,  \
          IF(T.ttime >= 10800 AND T.ttime < 21600, T.ping_count, 0) as 3_6,  \
          IF(T.ttime >= 21600 AND T.ttime < 32400, T.ping_count, 0) as 6_9, \
          IF(T.ttime >= 32400 AND T.ttime < 43200, T.ping_count, 0) as 9_12, \
          IF(T.ttime >= 43200 AND T.ttime < 54000, T.ping_count, 0) as 12_15, \
          IF(T.ttime >= 54000 AND T.ttime < 64800, T.ping_count, 0) as 15_18, \
          IF(T.ttime >= 64800 AND T.ttime < 75600, T.ping_count, 0) as 18_21, \
          IF(T.ttime >= 75600 AND T.ttime < 86400, T.ping_count, 0) as 21_24 \
          from ( \
          select us.user_id, TIME_TO_SEC(TIMEDIFF(us.started_at, user.created_at)) + 3600 as ttime, us.ping_count from user_session us \
            left join user ON user.id = us.user_id \
          where user.created_at > %s and TIME_TO_SEC(TIMEDIFF(us.started_at, user.created_at)) < 86400 and DATE(user.created_at) < DATE_SUB(CURDATE(), INTERVAL 2 DAY) and user.level_id != 2161) as T"

