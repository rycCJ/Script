# 这是初版代码，基于错误的PDF文件，课程安排不正确

import datetime
import uuid

# ==================== 数据区 (基于错误的PDF) ====================

# 学期第一周的周一日期
semester_start_date = datetime.date(2025, 9, 1)

# 课程信息 (这是不正确的版本)
courses = [
    {
        "name": "雷达测量与成像(校企课程)01",
        "day": "星期一",  # 错误的位置
        "start_time": "19:00:00",
        "end_time": "21:25:00",
        "weeks": "2-5",
        "location": "信远楼218",
        "instructor": "韩政胜"
    },
    {
        "name": "算法设计与分析01",
        "day": "星期二",  
        "start_time": "19:00:00",
        "end_time": "21:25:00",
        "weeks": "2-5,7-9",
        "location": "信远楼111",
        "instructor": "刘静, 赵宏"
    },
    {
        "name": "算法设计与分析01",
        "day": "星期四",
        "start_time": "08:30:00",
        "end_time": "12:00:00",
        "weeks": "2-4,6-9",
        "location": "信远楼111",
        "instructor": "刘静, 赵宏"
    },
    {
        "name": "图像视频理解与计算机视觉01",
        "day": "星期五",
        "start_time": "14:00:00",
        "end_time": "17:30:00", # PDF上的时间只到16:40
        "weeks": "2-4,6-10",
        "location": "信远楼218",
        "instructor": "宋丹"
    },
    {
        "name": "体育(羽毛球-02)",
        "day": "星期五",
        "start_time": "10:25:00",
        "end_time": "12:00:00",
        "weeks": "2-4,6-14",
        "location": "未指定",
        "instructor": "李秀"
    },
    {
        "name": "雷达测量与成像(校企课程)01",
        "day": "星期六",
        "start_time": "14:00:00",
        "end_time": "17:30:00", # PDF上的时间只到16:40
        "weeks": "2-4,6-7",
        "location": "信远楼218",
        "instructor": "韩政胜"
    }
]

# ===============================================================


# ==================== 核心逻辑区 (这部分是通用的) ====================

weekday_offset = {"星期一": 0, "星期二": 1, "星期三": 2, "星期四": 3, "星期五": 4, "星期六": 5, "星期日": 6}

def parse_week_ranges(week_str):
    weeks = set()
    parts = week_str.split(',')
    for part in parts:
        if '-' in part:
            start, end = map(int, part.split('-'))
            weeks.update(range(start, end + 1))
        else:
            weeks.add(int(part))
    return sorted(list(weeks))

def get_date_for_week_and_day(start_of_semester, week_num, day_str):
    days_to_add = (week_num - 1) * 7 + weekday_offset[day_str]
    return start_of_semester + datetime.timedelta(days=days_to_add)

def create_ics_event(course, start_week, end_week):
    start_date = get_date_for_week_and_day(semester_start_date, start_week, course['day'])
    until_date = get_date_for_week_and_day(semester_start_date, end_week, course['day'])
    
    start_time_obj = datetime.datetime.strptime(course['start_time'], '%H:%M:%S').time()
    end_time_obj = datetime.datetime.strptime(course['end_time'], '%H:%M:%S').time()

    dtstart = datetime.datetime.combine(start_date, start_time_obj)
    dtend = datetime.datetime.combine(start_date, end_time_obj)
    
    until_dt_utc = datetime.datetime.combine(until_date, datetime.time(23, 59, 59))
    
    dtstart_str = dtstart.strftime('%Y%m%dT%H%M%S')
    dtend_str = dtend.strftime('%Y%m%dT%H%M%S')
    until_str = until_dt_utc.strftime('%Y%m%dT%H%M%SZ')
    dtstamp_str = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    uid = uuid.uuid4()

    event_lines = [
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{dtstamp_str}",
        f"DTSTART;TZID=Asia/Shanghai:{dtstart_str}",
        f"DTEND;TZID=Asia/Shanghai:{dtend_str}",
        f"SUMMARY:{course['name']}",
        f"LOCATION:{course['location']}",
        f"DESCRIPTION:教师: {course['instructor']}",
    ]
    
    if start_week != end_week:
        event_lines.append(f"RRULE:FREQ=WEEKLY;UNTIL={until_str}")
    
    event_lines.append("END:VEVENT")
    return "\n".join(event_lines)

def main():
    ics_content = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//My Python Script//Schedule Generator//EN",
        "CALSCALE:GREGORIAN",
        "BEGIN:VTIMEZONE",
        "TZID:Asia/Shanghai",
        "BEGIN:STANDARD",
        "TZOFFSETFROM:+0800",
        "TZOFFSETTO:+0800",
        "TZNAME:CST",
        "DTSTART:19700101T000000",
        "END:STANDARD",
        "END:VTIMEZONE",
    ]

    for course in courses:
        all_weeks = parse_week_ranges(course['weeks'])
        if not all_weeks:
            continue

        week_groups = []
        if all_weeks:
            current_group = [all_weeks[0]]
            for i in range(1, len(all_weeks)):
                if all_weeks[i] == current_group[-1] + 1:
                    current_group.append(all_weeks[i])
                else:
                    week_groups.append(current_group)
                    current_group = [all_weeks[i]]
            week_groups.append(current_group)
        
        for group in week_groups:
            start_week = group[0]
            end_week = group[-1]
            ics_content.append(create_ics_event(course, start_week, end_week))

    ics_content.append("END:VCALENDAR")

    file_name = 'first_incorrect_schedule.ics'
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write("\n".join(ics_content))

    print(f"日历文件 '{file_name}' 已成功生成！")

if __name__ == "__main__":
    main()