# Padel Stats - MVP Walkthrough

## Summary
I have successfully initialized the **Padel Stats** project, establishing the core architecture for both Frontend and Backend, and implemented the initial set of features requested.

### 1. Project Structure
- **Monorepo**:
    - `/frontend`: Next.js 16 (React 19) + Tailwind CSS v4.
    - `/backend`: Python FastAPI + OpenCV + YOLOv8 (ready for integration).

### 2. Frontend Features
#### Player Stats Card (`StatsCard.tsx`)
A visual component to display player statistics with:
- **Circular Progress** for Win Rate.
- **Bar Charts** for individual shot performance (Forehand, Backhand, Volley, Smash).
- **Glassmorphism Design** consistent with the "Dark Mode" theme.

#### Video Upload (`VideoUpload.tsx`)
- Drag & Drop interface.
- Progress bar visualization.
- Connected to the backend API.

### 3. Backend Features
#### Upload API (`POST /upload`)
- Handles video file uploads.
- Generates unique IDs for each match.
- Saves files to `backend/uploaded_videos`.

## Verification Results

### Frontend UI
I have verified the UI runs correctly on `localhost:3000`.

**Dashboard (Landing Page) with Stats Cards:**
![Dashboard](file:///Users/miguelizaga/.gemini/antigravity/brain/44ab4368-c4c4-4101-9272-1a3ea7a1fc65/homepage_screenshot_1770412984673.png)

**Upload Page:**
![Upload Page](file:///Users/miguelizaga/.gemini/antigravity/brain/44ab4368-c4c4-4101-9272-1a3ea7a1fc65/upload_page_screenshot_1770412989012.png)

### Backend API
The backend is running at `http://localhost:8000`.
- **Health Check**: `GET /` returns `{"message": "Welcome to Padel Stats API"}` (Verified).
- **Upload**: Accepts video files and stores them successfully.

### AI Engine (YOLOv8)
I have successfully installed `ultralytics` and ran a test inference on a sample match video (`start-1.MOV`).
- **Input**: `backend/uploaded_videos/test.mov`
- **Output**: `runs/detect/predict/test.mp4`
- **Detections**: Properly identified players ("person") and the ball ("sports ball"). 

## Next Steps
1.  **Refine Tracking**: Assign unique IDs to players (already supported by YOLOv8 `track` mode).
2.  **Court Detection**: Implement logic to find court lines.
3.  **Video Player**: Implement a custom player to overlay analytics.
4.  **Database**: Connect a real database (SQLite/PostgreSQL) to store the data models permanently.
