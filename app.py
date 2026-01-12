import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose WhatsApp chat (.txt)")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    try:
        data = bytes_data.decode("utf-16")
    except UnicodeDecodeError:
        data = bytes_data.decode("utf-8", errors="ignore")

    df = preprocessor.preprocess(data)

    if df.empty:
        st.error("Chat file has no readable messages.")
        st.stop()

    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis for", user_list)

    if st.sidebar.button("Show Analysis"):

        # ✅ FIXED: UNPACK ONLY 4 VALUES
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(
            selected_user, df
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Messages", num_messages)
        col2.metric("Words", words)
        col3.metric("Media", num_media_messages)
        col4.metric("Links", num_links)

        if df.shape[0] >= 5:
            st.title("Monthly Timeline")
            timeline = helper.monthly_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'])
            plt.xticks(rotation=90)
            st.pyplot(fig)

            st.title("Daily Timeline")
            daily = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily['only_date'], daily['message'])
            plt.xticks(rotation=90)
            st.pyplot(fig)

        st.title("Activity Map")
        col1, col2 = st.columns(2)

        with col1:
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation=90)
            st.pyplot(fig)

        with col2:
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values)
            plt.xticks(rotation=90)
            st.pyplot(fig)

        if df.shape[0] >= 10:
            st.title("Weekly Activity Map")
            heatmap = helper.activity_heatmap(selected_user, df)
            fig, ax = plt.subplots()
            sns.heatmap(heatmap, ax=ax)
            st.pyplot(fig)

        st.title("Wordcloud")
        wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

        st.title("Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)

        if not emoji_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(emoji_df)

            with col2:
                fig, ax = plt.subplots()
                ax.pie(
                    emoji_df[1].head(),
                    labels=emoji_df[0].head(),
                    autopct="%0.2f"
                )
                st.pyplot(fig)
        else:
            st.info("No emojis found.")

        st.title("Emotion Intensity & Conversation Events")

        if df.shape[0] < 10:
            st.warning("Chat too small for emotion analysis.")
        else:
            emotion_df = helper.compute_emotion_intensity(selected_user, df)
            emotion_df = emotion_df.dropna()

            if emotion_df.empty:
                st.info("Not enough emotional variation.")
                st.stop()

            mean = emotion_df['emotion_intensity'].mean()
            std = emotion_df['emotion_intensity'].std()

            if std == 0 or np.isnan(std):
                st.info("Emotion variance too low.")
                st.stop()

            emotion_df['z'] = (emotion_df['emotion_intensity'] - mean) / std
            emotion_df['z_smooth'] = emotion_df['z'].rolling(12).mean()

            events_df = helper.detect_emotional_events(emotion_df)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(emotion_df['hour_block'], emotion_df['z_smooth'], color='black')
            ax.axhline(2, color='red', linestyle='--')
            ax.axhline(-2, color='green', linestyle='--')
            plt.xticks(rotation=90)
            st.pyplot(fig)

            if not events_df.empty:
                st.subheader("Detected Emotional Events")
                st.dataframe(events_df, use_container_width=True)
