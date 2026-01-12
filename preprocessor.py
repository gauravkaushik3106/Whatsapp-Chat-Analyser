import pandas as pd
import re

def preprocess(data):
    rows = []

    # Split by lines (most reliable)
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue

        # Match WhatsApp datetime prefix
        match = re.match(
            r'^(\d{1,2}/\d{1,2}/\d{2}),\s(\d{1,2}:\d{2})\s-\s(.*)',
            line
        )

        if match:
            date_part = match.group(1)
            time_part = match.group(2)
            message_part = match.group(3)

            rows.append({
                "raw_datetime": f"{date_part} {time_part}",
                "message": message_part
            })
        else:
            # Multiline message → append to previous
            if rows:
                rows[-1]["message"] += " " + line

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Convert datetime
    df['date'] = pd.to_datetime(
        df['raw_datetime'],
        format='%d/%m/%y %H:%M',
        errors='coerce'
    )

    df = df.dropna(subset=['date'])

    users = []
    messages = []

    for msg in df['message']:
        if ": " in msg:
            user, text = msg.split(": ", 1)
            users.append(user)
            messages.append(text)
        else:
            users.append("group_notification")
            messages.append(msg)

    df['user'] = users
    df['message'] = messages

    # Time features
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Period
    df['period'] = df['hour'].apply(
        lambda h: f"{h}-{(h+1)%24:02d}"
    )

    return df