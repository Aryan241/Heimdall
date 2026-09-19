import { NextRequest } from 'next/server';
import { readFile, stat } from 'fs/promises';
import path from 'path';
import { CONTENT_TYPES, jobPath } from '@/lib/server/heimdall';

export const runtime = 'nodejs';

/** Serve a job output file. Names are validated (no path separators, no dotfiles). */
export async function GET(_req: NextRequest, ctx: RouteContext<'/api/jobs/[id]/files/[name]'>) {
  const { id, name } = await ctx.params;
  let file: string;
  try {
    file = jobPath(id, name);
  } catch {
    return new Response('Bad request', { status: 400 });
  }
  try {
    const info = await stat(file);
    const data = await readFile(file);
    const ext = path.extname(name).toLowerCase();
    const download = ['.tif', '.glb', '.json', '.csv'].includes(ext) ? `attachment; filename="${name}"` : 'inline';
    return new Response(new Uint8Array(data), {
      headers: {
        'Content-Type': CONTENT_TYPES[ext] ?? 'application/octet-stream',
        'Content-Length': String(info.size),
        'Content-Disposition': download,
        'Cache-Control': 'private, max-age=3600',
      },
    });
  } catch {
    return new Response('Not found', { status: 404 });
  }
}
