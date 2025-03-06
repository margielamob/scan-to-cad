const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const cors = require('cors');

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