import { NextRequest } from 'next/server';
import { spawn } from 'child_process';
import { writeFile, copyFile, mkdir } from 'fs/promises';
import path from 'path';

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get('image') as File;
    const refDem = formData.get('reference_dem') as File | null;
    
    if (!file) {
      return new Response(JSON.stringify({ error: 'No image file provided' }), { status: 400 });
    }

    const buffer = Buffer.from(await file.arrayBuffer());
    
    const projectRoot = path.resolve(process.cwd(), '..');
    const dataDir = path.join(projectRoot, 'data');
    const publicModelsDir = path.join(process.cwd(), 'public', 'models');
    const publicDownloadsDir = path.join(process.cwd(), 'public', 'downloads');
    
    await mkdir(dataDir, { recursive: true });
    await mkdir(publicModelsDir, { recursive: true });
    await mkdir(publicDownloadsDir, { recursive: true });

    const inputFilename = `uploaded_image_${Date.now()}${path.extname(file.name)}`;
    const inputPath = path.join(dataDir, inputFilename);
    await writeFile(inputPath, buffer);

    let refDemPath = '';
    if (refDem) {
      const demBuffer = Buffer.from(await refDem.arrayBuffer());
      refDemPath = path.join(dataDir, `ref_dem_${Date.now()}${path.extname(refDem.name)}`);
      await writeFile(refDemPath, demBuffer);
    }

    const pythonExe = path.join(projectRoot, 'venv', 'bin', 'python');
    const inferScript = path.join(projectRoot, 'infer.py');
    const weightsPath = path.join(projectRoot, 'checkpoints', 'decoder', 'decoder_ep10_all.pth');
    
    const args = [inferScript, '--input', inputPath, '--export-mesh'];
    if (refDemPath) {
      args.push('--reference-dem', refDemPath, '--use-segmentation');
    } else {
      args.push('--weights', weightsPath);
    }

    // Set up SSE Stream
    const stream = new ReadableStream({
      start(controller) {
        const sendEvent = (event: string, data: any) => {
          controller.enqueue(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
        };

        sendEvent('status', { message: 'Starting pipeline...', stage: 1 });
        
        const child = spawn(pythonExe, args, { cwd: projectRoot });
        
        let outputText = '';

        child.stdout.on('data', (chunk) => {
          const text = chunk.toString();
          outputText += text;
          // Send raw logs for debugging/status updates
          sendEvent('log', { text });
          
          // Try to parse stage progress from logs
          if (text.includes('Stage 2')) sendEvent('status', { message: 'Extracting relative depth...', stage: 2 });
          if (text.includes('Stage 4') || text.includes('Stage 5') || text.includes('Stage 6')) {
            sendEvent('status', { message: 'Calibrating depth to absolute scale...', stage: 3 });
          }
          if (text.includes('Stage 8')) sendEvent('status', { message: 'Generating 3D mesh...', stage: 4 });
          
          // Parse metrics
          const metricsMatches = [...text.matchAll(/METRICS:(min|max|mean)=([\d.-]+)/g)];
          for (const match of metricsMatches) {
            sendEvent('metric', { key: match[1], value: parseFloat(match[2]) });
          }
        });

        child.stderr.on('data', (chunk) => {
          sendEvent('log', { text: chunk.toString(), isError: true });
        });

        child.on('close', async (code) => {
          if (code !== 0) {
            sendEvent('error', { message: `Pipeline failed with code ${code}` });
            controller.close();
            return;
          }

          try {
            const stem = path.parse(inputFilename).name;
            const outputMeshPath = path.join(projectRoot, 'outputs', `${stem}_mesh.glb`);
            const outputGeoTiffPath = path.join(projectRoot, 'outputs', `${stem}_dsm.tif`);
            const outputPngPath = path.join(projectRoot, 'outputs', `${stem}_depth16.png`);
            
            const timestamp = Date.now();
            const finalMeshName = `mesh_${timestamp}.glb`;
            const finalGeoTiffName = `dsm_${timestamp}.tif`;
            const finalPngName = `depth_${timestamp}.png`;
            
            const finalMeshPublicPath = path.join(publicModelsDir, finalMeshName);
            await copyFile(outputMeshPath, finalMeshPublicPath);
            
            try { await copyFile(outputGeoTiffPath, path.join(publicDownloadsDir, finalGeoTiffName)); } catch (e) {}
            try { await copyFile(outputPngPath, path.join(publicDownloadsDir, finalPngName)); } catch (e) {}

            sendEvent('done', {
              meshUrl: `/models/${finalMeshName}`,
              geotiff: `/downloads/${finalGeoTiffName}`,
              depthPng: `/downloads/${finalPngName}`
            });
            controller.close();
          } catch (err: any) {
            sendEvent('error', { message: err.message });
            controller.close();
          }
        });
      }
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });
    
  } catch (err: any) {
    return new Response(JSON.stringify({ error: err.message }), { status: 500 });
  }
}
