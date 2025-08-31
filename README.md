# Retrieval-Augmented Generation (RAG) for Indonesian Tax Law

A specialized chatbot application that provides accurate information about Indonesian tax statutory rules using Retrieval-Augmented Generation (RAG) technology.

## 🎯 Overview

This application combines the power of vector databases and large language models to create an intelligent legal assistant specifically designed for Indonesian tax law consultation. Users can ask questions in Indonesian language about tax regulations, and the system will retrieve relevant legal documents and provide accurate, contextual answers.

## 🏗️ Architecture

The application follows a modern RAG architecture with the following components:

### Backend (Python)
- **Vector Database**: Weaviate for storing and searching through legal document embeddings
- **Language Model**: OpenAI GPT-3.5-turbo for query optimization and response generation
- **Document Processing**: LangChain and Tika for PDF parsing and text chunking
- **API Layer**: Flask server handling chat requests and vector database operations

### Frontend (React)
- **Chat Interface**: Modern, responsive chat UI built with React
- **Styling**: TailwindCSS and DaisyUI for modern design
- **State Management**: React Context for chat state management

### Vector Database
- **Weaviate**: Self-hosted vector database with OpenAI embeddings
- **Document Storage**: Indonesian tax law documents stored as searchable vector embeddings
- **Semantic Search**: Natural language querying capabilities

## 🚀 Features

- **Indonesian Language Support**: Full support for Indonesian language queries and responses
- **Legal Expertise**: Specialized knowledge base of Indonesian tax statutory rules
- **Intelligent Query Processing**: Two-step RAG process for optimal results:
  1. Query optimization using GPT for better vector database searches
  2. Response generation based on retrieved relevant legal text
- **Document Processing**: Automatic processing and chunking of PDF legal documents
- **Real-time Chat**: Interactive chat interface for seamless user experience
- **Semantic Search**: Advanced vector similarity search for finding relevant legal provisions

## 📋 Prerequisites

- **Node.js** 14+ (for frontend)
- **Python** 3.8+ (for backend)  
- **Docker** and **Docker Compose** (for Weaviate database)
- **OpenAI API key** (required for the main functionality)
- **Prompt Optimization API key** (optional, for advanced query optimization)

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/atras30/retrieval-augmented-generation.git
cd retrieval-augmented-generation
```

### 2. Environment Setup

Create environment files with your API keys:

```bash
# Backend environment
cp server/.env.example server/.env

# Frontend environment (optional - for prompt optimization feature)
cp frontend/.env.example frontend/.env
```

Configure your backend environment variables:
```env
CHAT_GPT_API_KEY=your_openai_api_key_here
LOCAL_URL=http://localhost:8080
WEAVIATE_SCHEMA_NAME=Question
```

Configure your frontend environment variables (optional):
```env
REACT_APP_API_KEY=your_prompt_optimization_api_key_here
```

### 3. Start Weaviate Database
```bash
cd server
docker-compose up -d
```

### 4. Backend Setup
```bash
cd server
pip install -r requirements.txt  # Install Python dependencies

# First, populate the vector database with legal documents
python gpt.py  # Or call the populate function

# Then start the Flask API server
python skripsi.py
```

The API server will be available at http://localhost:5000

### 5. Frontend Setup
```bash
cd frontend
npm install
npm start
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000
- Weaviate Database: http://localhost:8080

## 📚 Usage

1. **Access the Chat Interface**: Open http://localhost:3000 in your browser
2. **Ask Legal Questions**: Type questions about Indonesian tax law in Indonesian language
3. **Get Expert Answers**: The system will:
   - Optimize your query for better document retrieval
   - Search through the legal document database
   - Generate accurate responses based on relevant legal provisions

### Example Queries
- "Apa itu PTKP?" (What is PTKP?)
- "Bagaimana cara menghitung tarif pajak penghasilan?" (How to calculate income tax rates?)
- "Apa saja peraturan terbaru tentang perpajakan?" (What are the latest tax regulations?)

## 🧪 Testing the Installation

### 1. Test Vector Database
```bash
# Check if Weaviate is running
curl http://localhost:8080/v1/meta

# Test semantic search directly
cd server
python semantic_search.py
```

### 2. Test API Server
```bash
# Test the populate endpoint
curl http://localhost:5000/populate

# Test a search query
curl "http://localhost:5000?search=apa itu pajak"
```

### 3. Test Command Line Interface
```bash
cd server
python gpt_chat.py  # Simple command-line chat for testing
```

## 📁 Project Structure

```
├── frontend/                 # React chat interface
│   ├── src/
│   │   ├── components/      # Chat components (ChatView, SideBar, etc.)
│   │   ├── context/         # React context for state management
│   │   └── App.js           # Main application component
│   ├── package.json
│   ├── .env.example         # Frontend environment configuration
│   └── README.md            # Frontend-specific documentation
├── server/                   # Python backend
│   ├── gpt.py              # Main RAG logic and GPT integration
│   ├── skripsi.py          # Flask API server
│   ├── rag.py              # Vector database population scripts
│   ├── semantic_search.py   # Search functionality
│   ├── gpt_chat.py         # Command-line chat interface
│   ├── docker-compose.yml   # Weaviate database setup
│   ├── requirements.txt     # Python dependencies
│   ├── .env.example         # Backend environment configuration
│   └── *.pdf               # Indonesian tax law documents
└── README.md               # This file
```

## 🔧 Key Components

### Backend Files
- **`gpt.py`**: Core RAG implementation with GPT integration
- **`skripsi.py`**: Flask API server handling HTTP requests
- **`rag.py`**: Vector database population and document processing
- **`semantic_search.py`**: Semantic search functionality
- **`gpt_chat.py`**: Simple command-line chat interface for testing
- **`docker-compose.yml`**: Weaviate database configuration

### Frontend Files
- **`App.js`**: Main React application
- **`ChatView.js`**: Chat interface component
- **`SideBar.js`**: Navigation sidebar component

## 📖 Legal Documents Included

The system includes several Indonesian tax law documents:
- UU Nomor 36 Tahun 2008 (Income Tax Law)
- UU Nomor 58 Tahun 2023 (Latest tax regulations with rate schemes)
- UU No 7 Tahun 2021 (Tax Law)

## 🤖 How RAG Works in This Application

1. **User Query**: User asks a question in Indonesian about tax law
2. **Query Optimization**: GPT optimizes the query for better vector database search
3. **Document Retrieval**: Weaviate searches for relevant legal document chunks
4. **Response Generation**: GPT generates an accurate answer based on retrieved legal text
5. **Indonesian Response**: User receives answer in Indonesian language

## 🧠 AI Roles

The system uses two specialized AI roles:
- **Vector Database Specialist**: Optimizes queries for better document retrieval
- **Legal Expert**: Provides accurate interpretations of Indonesian tax law

## ⚖️ Legal Disclaimer

This application is designed to assist with understanding Indonesian tax laws but should not be considered as official legal advice. Always consult with qualified tax professionals or official government sources for legal matters.

## 🔧 Troubleshooting

### Common Issues

**Weaviate Connection Error**
- Ensure Docker is running: `docker ps`
- Check if Weaviate is accessible: `curl http://localhost:8080/v1/meta`

**OpenAI API Error**
- Verify your API key is correctly set in `.env` files
- Check your OpenAI API quota and billing status

**Frontend Not Loading**
- Ensure the backend API is running on port 5000
- Check browser console for CORS errors
- Verify both frontend and backend environment variables are set

**Vector Database Empty**
- Run the populate endpoint: `curl http://localhost:5000/populate`
- Or manually run: `python server/gpt.py` (uncomment the populate_vector_database() call)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to improve the system.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.