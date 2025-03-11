const axios = require('axios');
const { createWriteStream } = require('fs');
const { promisify } = require('util');
const stream = require('stream');
const pipeline = promisify(stream.pipeline);
const AdmZip = require('adm-zip');
const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const cors = require('cors');
require('dotenv').config();
const Replicate = require('replicate');
// Create Express app
const app = express();
const port = process.env.PORT || 3000;

// Enable CORS
app.use(cors());

// Parse JSON bodies
app.use(express.json());

// Setup storage for uploaded files
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});

const upload = multer({ 
  storage: storage,
  limits: { fileSize: 100 * 1024 * 1024 }, // 100MB limit
  fileFilter: (req, file, cb) => {
    // Accept point clouds, meshes, and images
    const validTypes = ['.ply', '.stl', '.obj', '.xyz', '.pcd', '.jpg', '.png', '.tiff'];
    const ext = path.extname(file.originalname).toLowerCase();
    if (validTypes.includes(ext)) {
      return cb(null, true);
    }
    cb(new Error('Invalid file type. Supported formats: PLY, STL, OBJ, XYZ, PCD, JPG, PNG, TIFF'));
  }
});

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

// Serve processed CAD files
app.use('/output', express.static(path.join(__dirname, 'output')));

// API endpoints
app.post('/api/upload', upload.single('scanFile'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const inputFile = req.file.path;
    const outputDir = path.join(__dirname, 'output');
    
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    const outputFile = path.join(outputDir, `${path.parse(req.file.filename).name}.step`);
    
    // Process the file based on its type
    const fileType = path.extname(req.file.originalname).toLowerCase();
    
    // Choose processing pipeline based on file type
    let processingResult;
    if (['.jpg', '.png', '.tiff'].includes(fileType)) {
      processingResult = await processImageToCAD(inputFile, outputFile);
    } else {
      processingResult = await process3DFileToCAD(inputFile, outputFile);
    }
    
    res.json({
      message: 'File processed successfully',
      originalFile: req.file.filename,
      outputFile: path.basename(outputFile),
      downloadUrl: `/output/${path.basename(outputFile)}`,
      processingDetails: processingResult
    });
  } catch (error) {
    console.error('Error processing file:', error);
    res.status(500).json({ error: 'Error processing file', details: error.message });
  }
});

app.get('/api/files', (req, res) => {
  const outputDir = path.join(__dirname, 'output');
  
  if (!fs.existsSync(outputDir)) {
    return res.json({ files: [] });
  }
  
  const files = fs.readdirSync(outputDir)
    .filter(file => path.extname(file).toLowerCase() === '.step')
    .map(file => ({
      filename: file,
      url: `/output/${file}`,
      createdAt: fs.statSync(path.join(outputDir, file)).ctime
    }));
  
  res.json({ files });
});

// Add this new API endpoint to server.js
// Replace the existing Dreamcraft3D integration with this updated version

// Add to the top of server.js with other imports

// Then update the Dreamcraft3D endpoint
app.post('/api/dreamcraft', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image uploaded' });
    }

    // Check if it's an image file
    const fileType = path.extname(req.file.originalname).toLowerCase();
    if (!['.jpg', '.jpeg', '.png', '.webp'].includes(fileType)) {
      return res.status(400).json({ 
        error: 'Invalid file type. Only JPG, PNG, and WEBP images are supported for Dreamcraft3D.'
      });
    }

    // Get the prompt from request body
    const prompt = req.body.prompt || 'A detailed 3D model';
    const numSteps = parseInt(req.body.numSteps || '800');

    // Create a unique ID for this job
    const jobId = `dreamcraft-${Date.now()}`;
    const outputDir = path.join(__dirname, 'output');
    const zipPath = path.join(outputDir, `${jobId}.zip`);
    const modelDir = path.join(outputDir, jobId);
    
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    if (!fs.existsSync(modelDir)) {
      fs.mkdirSync(modelDir, { recursive: true });
    }

    // Send initial status response
    res.json({
      message: 'Dreamcraft3D processing started',
      jobId: jobId,
      status: 'processing'
    });

    // Create prediction
    console.log(`Starting Dreamcraft3D processing for job ${jobId}`);
    
    try {
      // Write status file to indicate processing has started
      fs.writeFileSync(
        path.join(outputDir, `${jobId}-status.json`), 
        JSON.stringify({
          jobId: jobId,
          status: 'processing',
          started: new Date().toISOString()
        })
      );
    
      // Initialize the Replicate client
      const replicate = new Replicate({
        auth: process.env.REPLICATE_API_TOKEN,
      });

      // Read the file and convert to base64
      const imageFile = fs.readFileSync(req.file.path);
      const base64Image = imageFile.toString('base64');
      const dataURI = `data:image/${fileType.substring(1)};base64,${base64Image}`;

      console.log("Running Dreamcraft3D model with the image and prompt...");
      
      // Run the model using the official client
      const model = "jd7h/dreamcraft3d";
      const version = "cf19b73a3c605ffa94c29d95971cb89823a0faa5f2ba830a3e1579fa61577c30";
      
      // Output will be a buffer or URL to the ZIP file
      const output = await replicate.run(
        `${model}:${version}`,
        {
          input: {
            image: dataURI,
            prompt: prompt, 
            num_steps: numSteps
          }
        }
      );

      console.log("Model run completed, downloading output...");
      
      // Check if output is a stream/buffer or a URL
      if (typeof output === 'string' && output.startsWith('http')) {
        // If it's a URL, download the file
        const response = await axios({
          method: 'GET',
          url: output,
          responseType: 'stream'
        });
        
        await pipeline(response.data, fs.createWriteStream(zipPath));
      } else {
        // If it's a buffer, write it directly
        fs.writeFileSync(zipPath, output);
      }
      
      console.log(`Downloaded ZIP file to ${zipPath}`);

      // Extract the ZIP file
      const zip = new AdmZip(zipPath);
      zip.extractAllTo(modelDir, true);
      
      // Find the OBJ file
      const objFile = fs.readdirSync(modelDir).find(file => file.endsWith('.obj'));
      
      if (!objFile) {
        throw new Error('No OBJ file found in the Dreamcraft3D output');
      }

      // Convert OBJ to STEP using our existing pipeline
      const objPath = path.join(modelDir, objFile);
      const stepPath = path.join(outputDir, `${jobId}.step`);
      
      const processingResult = await process3DFileToCAD(objPath, stepPath);
      
      // Update status in a job status file
      const statusData = {
        jobId: jobId,
        status: 'completed',
        objFile: objFile,
        stepFile: `${jobId}.step`,
        downloadUrl: `/output/${jobId}.step`,
        processingDetails: processingResult,
        completed: new Date().toISOString()
      };
      
      fs.writeFileSync(
        path.join(outputDir, `${jobId}-status.json`), 
        JSON.stringify(statusData)
      );
      
      console.log(`Processing completed for job ${jobId}`);
      
    } catch (error) {
      console.error('Error in Dreamcraft3D processing:', error);
      
      // Write error status
      fs.writeFileSync(
        path.join(outputDir, `${jobId}-status.json`), 
        JSON.stringify({
          jobId: jobId,
          status: 'failed',
          error: error.message || 'Unknown error',
          errorTime: new Date().toISOString()
        })
      );
    }
  } catch (error) {
    console.error('Error processing Dreamcraft3D request:', error);
    // If we haven't sent a response yet
    if (!res.headersSent) {
      res.status(500).json({ error: 'Error processing request', details: error.message });
    }
  }
});

// Add this endpoint to check the status of a Dreamcraft3D job
app.get('/api/dreamcraft/:jobId/status', (req, res) => {
  const { jobId } = req.params;
  const statusPath = path.join(__dirname, 'output', `${jobId}-status.json`);
  
  if (fs.existsSync(statusPath)) {
    const statusData = JSON.parse(fs.readFileSync(statusPath, 'utf8'));
    res.json(statusData);
  } else {
    res.json({ jobId, status: 'processing' });
  }
});

// Processing functions
async function processImageToCAD(inputFile, outputFile) {
  // Use our Python script with OpenCV and Open3D to convert an image to a 3D model
  const scriptPath = path.join(__dirname, 'scripts', 'image_to_3d.py');
  
  // Make sure the output directory exists
  const outputDir = path.dirname(outputFile);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  // Prepare parameters for the script
  const params = JSON.stringify({
    depth_method: 'sobel',
    downsample_factor: 4
  });
  
  return new Promise((resolve, reject) => {
    console.log(`Running image conversion: python ${scriptPath} ${inputFile} ${outputFile}`);
    
    // Spawn the Python process
    const process = spawn('python3', [scriptPath, inputFile, outputFile, params]);

    
    let stdoutData = '';
    let stderrData = '';
    
    process.stdout.on('data', (data) => {
      const dataStr = data.toString();
      stdoutData += dataStr;
      console.log(`Python stdout: ${dataStr}`);
    });
    
    process.stderr.on('data', (data) => {
      const dataStr = data.toString();
      stderrData += dataStr;
      console.error(`Python stderr: ${dataStr}`);
    });
    
    process.on('close', (code) => {
      if (code !== 0) {
        console.error(`Image processing failed with code ${code}`);
        console.error(`Error details: ${stderrData}`);
        return reject(new Error(`Image processing failed with code ${code}: ${stderrData}`));
      }
      
      try {
        // Try to parse the JSON result from the Python script
        // The last line of stdout should be the JSON result
        const resultLines = stdoutData.trim().split('\n');
        const resultJson = JSON.parse(resultLines[resultLines.length - 1]);
        
        if (!resultJson.success) {
          return reject(new Error(`Failed to convert image: ${resultJson.error || 'Unknown error'}`));
        }
        
        resolve({
          method: 'image-to-cad',
          steps: ['edge-detection', 'depth-estimation', 'model-generation', 'cad-conversion'],
          details: resultJson
        });
      } catch (error) {
        console.error('Error parsing Python script output:', error);
        reject(new Error(`Failed to parse conversion result: ${error.message}`));
      }
    });
  });
}

async function process3DFileToCAD(inputFile, outputFile) {
  // Use our Python script with Open3D to process the 3D file
  const scriptPath = path.join(__dirname, 'scripts', 'scan_to_cad.py');
  
  // Make sure the output directory exists
  const outputDir = path.dirname(outputFile);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  // Prepare parameters for the script
  const params = JSON.stringify({
    voxel_size: 0.01,
    plane_distance: 0.02,
    cylinder_distance: 0.01,
    min_points: 100
  });
  
  return new Promise((resolve, reject) => {
    console.log(`Running conversion: python ${scriptPath} ${inputFile} ${outputFile}`);
    
    // Spawn the Python process
    const process = spawn('python3', [scriptPath, inputFile, outputFile, params]);
    let stdoutData = '';
    let stderrData = '';
    
    process.stdout.on('data', (data) => {
      const dataStr = data.toString();
      stdoutData += dataStr;
      console.log(`Python stdout: ${dataStr}`);
    });
    
    process.stderr.on('data', (data) => {
      const dataStr = data.toString();
      stderrData += dataStr;
      console.error(`Python stderr: ${dataStr}`);
    });
    
    process.on('close', (code) => {
      if (code !== 0) {
        console.error(`Processing failed with code ${code}`);
        console.error(`Error details: ${stderrData}`);
        return reject(new Error(`Processing failed with code ${code}: ${stderrData}`));
      }
      
      try {
        // Try to parse the JSON result from the Python script
        // The last line of stdout should be the JSON result
        const resultLines = stdoutData.trim().split('\n');
        const resultJson = JSON.parse(resultLines[resultLines.length - 1]);
        
        if (!resultJson.success) {
          return reject(new Error(`Failed to convert file: ${resultJson.error || 'Unknown error'}`));
        }
        
        resolve({
          method: '3d-to-cad',
          steps: ['mesh-cleanup', 'feature-detection', 'surface-reconstruction', 'cad-conversion'],
          details: resultJson
        });
      } catch (error) {
        console.error('Error parsing Python script output:', error);
        reject(new Error(`Failed to parse conversion result: ${error.message}`));
      }
    });
  });
}

// Start server
app.listen(port, () => {
  console.log(`Scan to CAD server running on port ${port}`);
});

module.exports = app;