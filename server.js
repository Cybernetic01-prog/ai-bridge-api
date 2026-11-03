import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { createClient } from '@supabase/supabase-js';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json());
app.use(cors());

// Serve static frontend files from the 'public' folder
app.use(express.static(path.join(__dirname, 'public')));

const PORT = process.env.PORT || 5000;

// Initialize Supabase Client
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

// Initialize Google Gemini Free Tier for Vector Embeddings
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// Helper: Convert text into a numerical vector using Google's embedding model
async function getVectorEmbedding(text) {
  const response = await ai.models.embedContent({
    model: 'gemini-embedding-2',
    contents: text,
    config: {
      outputDimensionality: 768
    }
  });

  // Handle response structure from the updated @google/genai SDK (`response.embeddings`)
  if (response.embeddings && response.embeddings[0] && response.embeddings[0].values) {
    return response.embeddings[0].values;
  } else if (response.embedding && response.embedding.values) {
    return response.embedding.values;
  }
  
  throw new Error('Failed to extract embedding values from Gemini response');
}

// --- MIDDLEWARE: Verify Developer API Key ---
async function verifyApiKey(req, res, next) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Missing or malformed API key' });
  }

  const apiKey = authHeader.split(' ')[1];

  // Lookup API key in Supabase `api_keys` table
  const { data, error } = await supabase
    .from('api_keys')
    .select('user_id, status')
    .eq('key', apiKey)
    .single();

  if (error || !data || data.status !== 'active') {
    return res.status(403).json({ error: 'Invalid, revoked, or unpaid API key' });
  }

  req.userId = data.user_id; // Pass user ID down the line
  next();
}

// --- ROUTE 1: Healthcheck ---
app.get('/api/health', (req, res) => {
  res.json({ status: 'AI Bridge API is online 🚀' });
});

// --- ROUTE 2: Add/Store Memory (Handles both /v1/memory/add and /v1/memory/store) ---
const handleAddMemory = async (req, res) => {
  try {
    const { text, metadata } = req.body;
    if (!text) return res.status(400).json({ error: 'Memory text is required' });

    // Generate real vector embedding
    const embeddingVector = await getVectorEmbedding(text);

    // Insert into Supabase `memories` table
    const { data, error } = await supabase
      .from('memories')
      .insert([
        { 
          user_id: req.userId, 
          content: text, 
          embedding: embeddingVector,
          metadata: metadata || {} 
        }
      ])
      .select();

    if (error) throw error;

    res.status(201).json({ 
      success: true, 
      message: 'Memory encrypted and saved to vault', 
      memory_id: data[0].id 
    });

  } catch (err) {
    console.error('Add Memory Error:', err);
    res.status(500).json({ error: err.message || 'Internal server error saving memory' });
  }
};

app.post('/v1/memory/add', verifyApiKey, handleAddMemory);
app.post('/v1/memory/store', verifyApiKey, handleAddMemory);

// --- ROUTE 3: Search Memory (POST /v1/memory/search) ---
app.post('/v1/memory/search', verifyApiKey, async (req, res) => {
  try {
    const { query, limit = 3 } = req.body;
    if (!query) return res.status(400).json({ error: 'Search query is required' });

    // Convert search query into vector
    const queryVector = await getVectorEmbedding(query);

    // Run cosine similarity search in Supabase
    const { data: memories, error } = await supabase.rpc('match_memories', {
      query_embedding: queryVector,
      match_threshold: 0.3,
      match_count: limit,
      p_user_id: req.userId
    });

    if (error) throw error;

    res.json({ 
      success: true, 
      count: memories.length,
      memories: memories.map(m => ({ id: m.id, text: m.content, metadata: m.metadata, score: m.similarity })) 
    });

  } catch (err) {
    console.error('Search Memory Error:', err);
    res.status(500).json({ error: err.message || 'Internal server error searching memory' });
  }
});

// Start API Server
app.listen(PORT, () => {
  console.log(`AI Bridge Server running on http://localhost:${PORT}`);
});