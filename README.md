# SizeSync — AI-Powered 3D Body Sizing

## Features

- Upload front, side, and back body images
- Enter user height for body scale reference
- Validate image quality and body framing
- Detect blurry, overexposed, underexposed, or low-resolution images
- Detect people and multiple-person images
- Check image framing and cropping
- Store user profiles and body profile records
- View image validation results
- FastAPI REST API
- React and TypeScript frontend

## Tech Stack

**Frontend**

- React
- TypeScript
- Tailwind CSS

**Backend**

- Python
- FastAPI
- SQLAlchemy
- SQLite
- OpenCV

**Planned AI/3D Technologies**

- MediaPipe Pose
- SMPL-X
- 3D mesh generation
- GLB model export

## Project Structure

```text
SizeSync/
├── backend/
│   ├── app/
│   ├── requirements.txt
│   └── README.md
├── frontend/
│   └── README.md
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js and npm
- Git

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd SizeSync
```

### 2. Set Up the Backend

```bash
cd backend
python -m venv .venv
```

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the environment file if the project includes `.env.example`:

```bash
copy .env.example .env
```

Start the backend server:

```bash
uvicorn app.main:app --reload --port 8000
```

API documentation:

[http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Set Up the Frontend

Open a new terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the frontend:

[http://localhost:5173](http://localhost:5173)

## API Endpoints

| Method | Endpoint             | Description                              |
| ------ | -------------------- | ---------------------------------------- |
| POST   | `/api/profiles`      | Create a profile and submit profile data |
| GET    | `/api/profiles/{id}` | Retrieve a profile by ID                 |

## Development Roadmap

      Frontend profile creation interface
      Body image upload workflow
      Image quality validation
      Backend API and database models
      Reconstruction service interface
      Image preprocessing
      Body segmentation and pose estimation
      Multi-view 3D body reconstruction
      SMPL-X body fitting
      3D mesh export
      Body measurement estimation
      Measurement confidence scoring
      Body profile dashboard
      Virtual garment fitting
      Personalized clothing size recommendations

