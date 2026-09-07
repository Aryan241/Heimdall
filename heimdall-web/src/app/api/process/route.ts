import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { writeFile, copyFile, mkdir } from 'fs/promises';
import path from 'path';
import util from 'util';

const execPromise = util.promisify(exec);

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get('image') as File;
    
    if (!file) {
      return NextResponse.json({ error: 'No image file provided' }, { status: 400 });
    }

    const buffer = Buffer.from(await file.arrayBuffer());
    
    // Determine absolute paths. Assuming nextjs app is in Heimdall/heimdall-web
    // and python backend is in Heimdall/
    const projectRoot = path.resolve(process.cwd(), '..');
    const dataDir = path.join(projectRoot, 'data');
    const publicModelsDir = path.join(process.cwd(), 'public', 'models');
    
    // Ensure directories exist
    await mkdir(dataDir, { recursive: true });
    await mkdir(publicModelsDir, { recursive: true });

    const inputFilename = 'uploaded_image.jpg';
    const inputPath = path.join(dataDir, inputFilename);
    
    // Save the uploaded file
    await writeFile(inputPath, buffer);
    console.log(`Saved uploaded image to ${inputPath}`);

    // Execute Python script
    const pythonExe = path.join(projectRoot, 'venv', 'bin', 'python');
    const inferScript = path.join(projectRoot, 'infer.py');
    const weightsPath = path.join(projectRoot, 'checkpoints', 'decoder', 'decoder_ep10_all.pth');
    
    const command = `"${pythonExe}" "${inferScript}" --input "${inputPath}" --weights "${weightsPath}" --export-mesh`;
    
    console.log(`Running pipeline: ${command}`);
    
    // Set maxBuffer large enough to capture python logs
    const { stdout, stderr } = await execPromise(command, { cwd: projectRoot, maxBuffer: 1024 * 1024 * 10 });
    
    console.log('Python output:', stdout);
    if (stderr) console.warn('Python stderr:', stderr);

    // After success, copy the generated mesh to the Next.js public directory
    const outputMeshName = 'uploaded_image_mesh.ply';
    const outputMeshPath = path.join(projectRoot, 'outputs', outputMeshName);
    
    const timestamp = Date.now();
    const finalMeshName = `mesh_${timestamp}.ply`;
    const finalPublicPath = path.join(publicModelsDir, finalMeshName);
    
    await copyFile(outputMeshPath, finalPublicPath);
    console.log(`Copied output mesh to ${finalPublicPath}`);

    return NextResponse.json({ 
      success: true, 
      meshUrl: `/models/${finalMeshName}`,
      message: 'Pipeline completed successfully'
    });

  } catch (error: any) {
    console.error('Pipeline Error:', error);
    return NextResponse.json({ 
      error: 'Failed to process image', 
      details: error.message 
    }, { status: 500 });
  }
}
