import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls
from docx.opc.constants import RELATIONSHIP_TYPE

def set_cell_shading(cell, color_hex):
    """Set background color of a table cell."""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set inner margins (padding) for a table cell in dxa (1 pt = 20 dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for margin_name, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{margin_name}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_hyperlink(paragraph, url, text, color_hex="2563EB", underline=True, bold=False, font_size_pt=9.0):
    """Add a real clickable hyperlink to a paragraph with proper formatting."""
    part = paragraph.part
    r_id = part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'), r_id)
    
    new_run = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')
    
    rFonts = OxmlElement('w:rFonts')
    rFonts.set(qn('w:ascii'), 'Calibri')
    rFonts.set(qn('w:hAnsi'), 'Calibri')
    rPr.append(rFonts)
    
    if color_hex:
        c = OxmlElement('w:color')
        c.set(qn('w:val'), color_hex)
        rPr.append(c)
        
    if underline:
        u = OxmlElement('w:u')
        u.set(qn('w:val'), 'single')
        rPr.append(u)
        
    if bold:
        b = OxmlElement('w:b')
        rPr.append(b)

    if font_size_pt:
        sz = OxmlElement('w:sz')
        sz.set(qn('w:val'), str(int(font_size_pt * 2)))
        rPr.append(sz)
        
    new_run.append(rPr)
    t = OxmlElement('w:t')
    t.text = text
    new_run.append(t)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)

def create_callout_box(doc, title_text, body_text, border_color="0D9488", bg_color="F0FDFA"):
    """Create a stylized callout box with a colored left accent border."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_shading(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=180, right=160)
    
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>\n'
        f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>\n'
        f'  <w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>\n'
        f'  <w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>\n'
        f'</w:tcBorders>'
    )
    tcPr.append(tcBorders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run_title = p.add_run(f"KEY CONCEPT: {title_text}\n")
    run_title.bold = True
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(11)
    run_title.font.color.rgb = RGBColor(13, 148, 136)
    
    run_body = p.add_run(body_text)
    run_body.font.name = "Calibri"
    run_body.font.size = Pt(10)
    run_body.font.color.rgb = RGBColor(31, 41, 55)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def add_code_block(doc, code_str, language_label="Python Blueprint"):
    """Add a stylized code block with monospaced font and subtle framing."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_shading(cell, "F8FAFC")
    set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
    
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="single" w:sz="6" w:space="0" w:color="E2E8F0"/>\n'
        f'  <w:left w:val="single" w:sz="18" w:space="0" w:color="3B82F6"/>\n'
        f'  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="E2E8F0"/>\n'
        f'  <w:right w:val="single" w:sz="6" w:space="0" w:color="E2E8F0"/>\n'
        f'</w:tcBorders>'
    )
    tcPr.append(tcBorders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.05
    
    tag_run = p.add_run(f"// {language_label}\n")
    tag_run.font.name = "Consolas"
    tag_run.font.size = Pt(8.0)
    tag_run.font.color.rgb = RGBColor(100, 116, 139)
    tag_run.italic = True
    
    run = p.add_run(code_str)
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(15, 23, 42)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def style_heading(heading, font_name="Calibri", font_size=16, color_rgb=(30, 58, 138), space_before=14, space_after=6):
    heading.paragraph_format.space_before = Pt(space_before)
    heading.paragraph_format.space_after = Pt(space_after)
    for r in heading.runs:
        r.font.name = font_name
        r.font.size = Pt(font_size)
        r.bold = True
        r.font.color.rgb = RGBColor(*color_rgb)

def add_bullet_with_bold_prefix(doc, prefix, body):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    run_prefix = p.add_run(prefix + ": ")
    run_prefix.bold = True
    run_prefix.font.name = "Calibri"
    run_prefix.font.size = Pt(10.5)
    run_prefix.font.color.rgb = RGBColor(30, 58, 138)
    
    run_body = p.add_run(body)
    run_body.font.name = "Calibri"
    run_body.font.size = Pt(10.5)
    run_body.font.color.rgb = RGBColor(55, 65, 81)
    return p

def add_styled_paragraph(doc, text, bold=False, italic=False, space_after=6, color_rgb=(55, 65, 81)):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.name = "Calibri"
    run.font.size = Pt(10.5)
    run.font.color.rgb = RGBColor(*color_rgb)
    return p

def create_resource_table(doc, headers, rows, widths, link_label="Open Resource"):
    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    # Header row
    for idx, text in enumerate(headers):
        cell = table.cell(0, idx)
        cell.width = widths[idx]
        set_cell_shading(cell, "1E3A8A")
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        p = cell.paragraphs[0]
        r = p.add_run(text)
        r.bold = True
        r.font.name = "Calibri"
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(255, 255, 255)
        
    for row_idx, row_data in enumerate(rows, start=1):
        shading = "F8FAFC" if row_idx % 2 == 1 else "FFFFFF"
        for c_idx, width in enumerate(widths):
            cell = table.cell(row_idx, c_idx)
            cell.width = width
            set_cell_shading(cell, shading)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            
        # Topic / Name
        p0 = table.cell(row_idx, 0).paragraphs[0]
        r0 = p0.add_run(row_data[0])
        r0.bold = True
        r0.font.name = "Calibri"; r0.font.size = Pt(9.5); r0.font.color.rgb = RGBColor(30, 58, 138)
        
        # Description
        p1 = table.cell(row_idx, 1).paragraphs[0]
        r1 = p1.add_run(row_data[1])
        r1.font.name = "Calibri"; r1.font.size = Pt(9.0); r1.font.color.rgb = RGBColor(55, 65, 81)
        
        # Link
        p2 = table.cell(row_idx, 2).paragraphs[0]
        add_hyperlink(p2, row_data[2], link_label, color_hex="2563EB", underline=True, bold=False, font_size_pt=9.0)
        
    doc.add_paragraph().paragraph_format.space_after = Pt(10)

def main():
    doc = docx.Document()
    
    # Page Margins (1.0 inch all around)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Document Title Block
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(10)
    title_p.paragraph_format.space_after = Pt(2)
    title_run = title_p.add_run("Heimdall Engineering Curriculum")
    title_run.bold = True
    title_run.font.name = "Calibri"
    title_run.font.size = Pt(26)
    title_run.font.color.rgb = RGBColor(30, 58, 138)
    
    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_before = Pt(0)
    sub_p.paragraph_format.space_after = Pt(12)
    sub_run = sub_p.add_run("Master Roadmap: Building Monocular Satellite 3D Digital Surface Reconstruction From Scratch")
    sub_run.font.name = "Calibri"
    sub_run.font.size = Pt(13)
    sub_run.font.color.rgb = RGBColor(13, 148, 136)
    
    # Metadata Overview Table
    meta_table = doc.add_table(rows=2, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    col_widths = [Inches(3.25), Inches(3.25)]
    for r in meta_table.rows:
        for idx, width in enumerate(col_widths):
            r.cells[idx].width = width
            set_cell_shading(r.cells[idx], "F8FAFC")
            set_cell_margins(r.cells[idx], top=70, bottom=70, left=110, right=110)
            
    r0c0 = meta_table.cell(0, 0).paragraphs[0]
    r0c0.add_run("Core Domains: ").bold = True
    r0c0.add_run("Deep Learning, 3D Vision, Geospatial & WebGIS")
    r0c0.runs[0].font.size = Pt(9.5); r0c0.runs[1].font.size = Pt(9.5)
    
    r0c1 = meta_table.cell(0, 1).paragraphs[0]
    r0c1.add_run("Target Deliverable: ").bold = True
    r0c1.add_run("2D GeoTIFF -> Metric DSM Raster + 3D Mesh GLB")
    r0c1.runs[0].font.size = Pt(9.5); r0c1.runs[1].font.size = Pt(9.5)
    
    r1c0 = meta_table.cell(1, 0).paragraphs[0]
    r1c0.add_run("Target Audience: ").bold = True
    r1c0.add_run("Software Engineers, AI Researchers, WebGIS Developers")
    r1c0.runs[0].font.size = Pt(9.5); r1c0.runs[1].font.size = Pt(9.5)
    
    r1c1 = meta_table.cell(1, 1).paragraphs[0]
    r1c1.add_run("Included Resources: ").bold = True
    r1c1.add_run("Landmark Papers, YouTube Lectures, Code Blueprints")
    r1c1.runs[0].font.size = Pt(9.5); r1c1.runs[1].font.size = Pt(9.5)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    
    # -------------------------------------------------------------
    # 1. EXECUTIVE OVERVIEW
    # -------------------------------------------------------------
    h1 = doc.add_heading("1. Executive Overview: What Does It Take to Build This?", level=1)
    style_heading(h1, font_size=16, color_rgb=(30, 58, 138), space_before=14, space_after=6)
    
    add_styled_paragraph(
        doc,
        "The Heimdall system is an end-to-end production AI and geospatial pipeline that converts a single monocular optical satellite RGB image into a metric 3D Digital Surface Model (DSM) and an interactive 3D WebGL terrain flythrough in real time. Unlike traditional stereo photogrammetry (which requires multi-angle satellite passes and intensive pixel matching) or airborne LiDAR surveys (which cost tens of thousands of dollars per flight), Heimdall computes metric elevations from single-pass imagery."
    )
    add_styled_paragraph(
        doc,
        "To build this system completely from scratch by yourself, you must integrate five distinct technical disciplines:"
    )
    
    add_bullet_with_bold_prefix(doc, "1. Deep Learning & Vision Transformers", "Vision Transformers (ViT-Large/Giant), self-supervised geometric feature representations (DINOv2), dense prediction transformers (DPT), and monocular metric depth architectures.")
    add_bullet_with_bold_prefix(doc, "2. Geospatial Science & Geodesy", "Coordinate Reference Systems (CRS/EPSG), geodetic datums, UTM metric projections, Ground Sampling Distance (GSD), affine geotransforms, and raster manipulation via GDAL/Rasterio.")
    add_bullet_with_bold_prefix(doc, "3. 3D Computer Vision & Geometry", "Pinhole camera models, back-projection formulas (2D pixel coordinates to 3D Cartesian space), surface normal derivation, point cloud filtering, and grid-based heightfield mesh triangulation into binary glTF/GLB.")
    add_bullet_with_bold_prefix(doc, "4. Custom Decoder Design & Loss Engineering", "Atrous Spatial Pyramid Pooling (ASPP), Scale-Invariant Logarithmic (SILog) loss, Sobel edge-aware gradient loss, normal consistency constraints, and mixed-precision (AMP) training.")
    add_bullet_with_bold_prefix(doc, "5. Interactive 3D WebGIS Engineering", "Next.js 14 App Router, React Three Fiber (R3F), WebGL shaders, camera clipping and atmospheric fog tuning, elevation colormaps, and high-speed binary mesh streaming.")
    
    create_callout_box(
        doc,
        "THE MONOCULAR DEPTH PARADOX",
        "Estimating 3D height from a single 2D image is an ill-posed inverse problem: infinitely many 3D scenes can project to the exact same 2D pixel array. Solving this requires deep neural priors learned across millions of images combined with rigorous camera unprojection math and spatial loss constraints."
    )

    # -------------------------------------------------------------
    # 2. DEEP LEARNING & VISION TRANSFORMERS
    # -------------------------------------------------------------
    h1 = doc.add_heading("2. Deep Learning Foundations & Vision Transformers", level=1)
    style_heading(h1, font_size=16, color_rgb=(30, 58, 138), space_before=16, space_after=6)
    
    add_styled_paragraph(
        doc,
        "Standard Convolutional Neural Networks (CNNs) struggle with wide-area satellite scenes because convolutions operate within small local receptive fields (3x3 or 5x5 kernels). Satellite elevation reconstruction requires correlating distant building shadows, sun elevation angles, mountain ridgelines, and regional context across thousands of pixels. Vision Transformers (ViTs) solve this via global self-attention."
    )
    
    h2 = doc.add_heading("Key Theoretical Concepts", level=2)
    style_heading(h2, font_size=12, color_rgb=(13, 148, 136), space_before=6, space_after=4)
    
    add_bullet_with_bold_prefix(doc, "PyTorch Computational Graph & Memory Optimization", "Autograd mechanisms, tensor memory layouts (channels-first vs. channels-last), CUDA stream synchronization, and custom torch.utils.data.Dataset / DataLoader pipelines.")
    add_bullet_with_bold_prefix(doc, "Patchify & Self-Attention Math", "Splitting an image of shape (H, W, 3) into non-overlapping patches of size P x P. Linear projection to embedding dimension D. Multi-head self-attention: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V.")
    add_bullet_with_bold_prefix(doc, "Self-Supervised Geometric Backbones (DINOv2)", "Why DINOv2 is the backbone of modern dense vision: trained without human labels on 142M curated images using student-teacher self-distillation. It learns dense, patch-level geometric descriptors that generalize universally across aerial, satellite, and ground views.")
    add_bullet_with_bold_prefix(doc, "Dense Prediction Transformers (DPT)", "Extracting feature tokens from multiple transformer stages (e.g., layers 4, 11, 17, 23). Transforming 1D sequence tokens back into 2D spatial feature maps via Reassemble and Fusion blocks.")
    add_bullet_with_bold_prefix(doc, "Resolving Scale and Shift Ambiguity", "Relative depth models output an uncalibrated disparity d_rel. The true metric depth relates via d_metric = s * d_rel + t, where s is scale and t is shift. Foundation models (Depth Anything V3, Metric3D) learn to resolve s and t using camera intrinsics and learned metric priors.")
    
    add_styled_paragraph(doc, "Below is the core PyTorch blueprint for extracting intermediate ViT tokens and reassembling them into multi-scale 2D feature maps:")
    
    vit_code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        "class ViTFeatureExtractor(nn.Module):\n"
        "    \"\"\"Extract intermediate token representations from Vision Transformer.\"\"\"\n"
        "    def __init__(self, vit_backbone, hook_layers=[4, 11, 17, 23]):\n"
        "        super().__init__()\n"
        "        self.backbone = vit_backbone\n"
        "        self.hook_layers = hook_layers\n"
        "        self.features = {}\n"
        "        for layer_idx in hook_layers:\n"
        "            self.backbone.blocks[layer_idx].register_forward_hook(\n"
        "                self._make_hook(layer_idx)\n"
        "            )\n"
        "\n"
        "    def _make_hook(self, layer_idx):\n"
        "        def hook(module, input, output):\n"
        "            # output shape: (B, N_patches + 1, D_embed)\n"
        "            self.features[layer_idx] = output[:, 1:, :]  # drop [CLS] token\n"
        "        return hook\n"
        "\n"
        "    def forward(self, x):\n"
        "        self.features.clear()\n"
        "        _ = self.backbone(x)\n"
        "        return self.features\n"
    )
    add_code_block(doc, vit_code, "PyTorch ViT Token Extractor Blueprint")

    table_widths = [Inches(1.8), Inches(2.7), Inches(2.0)]
    headers = ["Topic / Resource", "What You Will Learn", "Link / Reference"]
    dl_rows = [
        ("Stanford CS231n: Deep Learning for Computer Vision", "The premier university course covering CNNs, spatial transformers, optimization, backpropagation, and dense prediction.", "https://cs231n.stanford.edu/"),
        ("3Blue1Brown: Neural Networks & Transformers", "The finest visual and mathematical intuition for neural networks, gradient descent, and attention mechanisms.", "https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi"),
        ("Vision Transformer (ViT) Paper (ICLR 2021)", "The seminal paper: 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'.", "https://arxiv.org/abs/2010.11929"),
        ("DINOv2: Robust Visual Features (Meta AI)", "The foundation for visual geometry and zero-shot dense feature extraction without human labels.", "https://arxiv.org/abs/2304.07193"),
        ("Depth Anything: Unleashing Large-Scale Unlabeled Data", "The architecture that powers modern monocular depth estimation, detailed training strategies, and scaling laws.", "https://arxiv.org/abs/2401.10891")
    ]
    create_resource_table(doc, headers, dl_rows, table_widths, link_label="Open Resource")

    # -------------------------------------------------------------
    # 3. GEOSPATIAL SCIENCE & GEODESY
    # -------------------------------------------------------------
    h1 = doc.add_heading("3. Geospatial Science, Remote Sensing & Geodesy", level=1)
    style_heading(h1, font_size=16, color_rgb=(30, 58, 138), space_before=16, space_after=6)
    
    add_styled_paragraph(
        doc,
        "Machine learning practitioners frequently fail when working with satellite data because they treat satellite imagery like standard JPEG photos. Satellite imagery is physically situated in space, tied to ellipsoids, datums, and projected coordinate reference systems."
    )
    
    h2 = doc.add_heading("Key Theoretical Concepts", level=2)
    style_heading(h2, font_size=12, color_rgb=(13, 148, 136), space_before=6, space_after=4)
    
    add_bullet_with_bold_prefix(doc, "Geodetic Datums & CRS (EPSG:4326 vs EPSG:326xx)", "Geographic coordinates (EPSG:4326 WGS84) measure angles in degrees (latitude/longitude). Because degrees vary in physical meter distance across latitudes, 3D metric elevation is strictly invalid in EPSG:4326. You must reproject to UTM (Universal Transverse Mercator, EPSG:326xx), where X, Y, and Z are all measured uniformly in meters.")
    add_bullet_with_bold_prefix(doc, "Ground Sampling Distance (GSD)", "The real-world physical ground distance covered by a single pixel (e.g., 0.5m GSD means 1 pixel = 50cm x 50cm on Earth). GSD dictates the horizontal scale factor required to convert pixel coordinates into physical spatial dimensions.")
    add_bullet_with_bold_prefix(doc, "DEM vs. DTM vs. DSM", "A Digital Elevation Model (DEM) is the umbrella term. A Digital Terrain Model (DTM) represents bare-Earth ground elevation (vegetation and man-made structures stripped away). A Digital Surface Model (DSM) represents the top-most reflective surface including tree canopies, rooftops, and towers. Heimdall reconstructs the DSM.")
    add_bullet_with_bold_prefix(doc, "Affine Geotransform Matrix", "Translates image raster row/col indices into real-world geographic coordinates: X_geo = A * col + B * row + C, Y_geo = D * col + E * row + F. Preserving this transform is mandatory for GIS interoperability.")
    add_bullet_with_bold_prefix(doc, "The Geospatial Stack (Rasterio, GDAL & PyProj)", "Reading multi-band satellite rasters, writing 32-bit floating-point elevation arrays, maintaining spatial extents, handling NoData masks, and exporting Cloud-Optimized GeoTIFFs (COG).")
    
    add_styled_paragraph(doc, "Here is how to write a predicted DSM array into a valid, georeferenced single-band 32-bit float GeoTIFF with preserved CRS and affine metadata:")
    
    raster_code = (
        "import rasterio\n"
        "from rasterio.crs import CRS\n"
        "\n"
        "def save_georeferenced_dsm(dsm_array, ref_geotiff_path, out_dsm_path):\n"
        "    \"\"\"Export predicted elevation array as compliant 32-bit GeoTIFF.\"\"\"\n"
        "    with rasterio.open(ref_geotiff_path) as src:\n"
        "        meta = src.meta.copy()\n"
        "        transform = src.transform\n"
        "        crs = src.crs\n"
        "\n"
        "    meta.update({\n"
        "        'driver': 'GTiff',\n"
        "        'count': 1,              # Single elevation band\n"
        "        'dtype': 'float32',        # Metric height in meters\n"
        "        'nodata': -9999.0,\n"
        "        'compress': 'deflate',\n"
        "        'transform': transform,\n"
        "        'crs': crs\n"
        "    })\n"
        "\n"
        "    with rasterio.open(out_dsm_path, 'w', **meta) as dst:\n"
        "        dst.write(dsm_array.astype('float32'), 1)\n"
    )
    add_code_block(doc, raster_code, "Rasterio Georeferenced DSM Export Blueprint")

    geo_rows = [
        ("NASA ARSET: Remote Sensing Fundamentals", "Free NASA training on satellite orbits, sensors, spectral resolutions, and elevation modeling.", "https://appliedsciences.nasa.gov/what-we-do/capacity-building/arset"),
        ("Rasterio: Geospatial Raster I/O for Python", "The definitive Python documentation for reading, writing, and transforming georeferenced raster data.", "https://rasterio.readthedocs.io/en/stable/"),
        ("Spatial Reference Database (EPSG.io)", "Searchable catalog of all global coordinate reference systems, UTM zones, and projection parameters.", "https://epsg.io/"),
        ("QGIS Official Training Manual & Tutorials", "Mastering the open-source GIS workbench for visually validating, contouring, and hillshading GeoTIFFs.", "https://docs.qgis.org/latest/en/docs/training_manual/"),
        ("GDAL: Geospatial Data Abstraction Library", "The C++/Python foundation that powers all modern geospatial tools and satellite imagery pipelines.", "https://gdal.org/")
    ]
    create_resource_table(doc, headers, geo_rows, table_widths, link_label="Open Resource")

    # -------------------------------------------------------------
    # 4. 3D COMPUTER VISION & PROJECTIVE GEOMETRY
    # -------------------------------------------------------------
    h1 = doc.add_heading("4. 3D Computer Vision, Projective Geometry & Meshing", level=1)
    style_heading(h1, font_size=16, color_rgb=(30, 58, 138), space_before=16, space_after=6)
    
    add_styled_paragraph(
        doc,
        "A depth map is only a 2D matrix of numbers. To render it in 3D WebGL, export it to CAD, or inspect it in GIS tools, you must unproject every pixel into 3D Cartesian space (X, Y, Z), compute surface normal vectors, and triangulate vertices into 3D polygon meshes."
    )
    
    h2 = doc.add_heading("Key Theoretical Concepts", level=2)
    style_heading(h2, font_size=12, color_rgb=(13, 148, 136), space_before=6, space_after=4)
    
    add_bullet_with_bold_prefix(doc, "Pinhole Camera Intrinsics (K Matrix)", "The intrinsic matrix K parameterizes the perspective projection: K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]. For satellite imagery, orthorectification simulates an orthographic or high-altitude equivalent perspective camera with long focal length.")
    add_bullet_with_bold_prefix(doc, "Back-Projection Mathematics", "Every 2D pixel (u, v) with metric depth Z is unprojected to 3D Cartesian camera coordinates using the inverse intrinsic formulation: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy, Z = depth(u, v).")
    add_bullet_with_bold_prefix(doc, "Coordinate Frame Conversions", "Bridging the gap between OpenCV (X right, Y down, Z forward), WebGL/Three.js (X right, Y up, Z backward), and Geospatial ENU (East-X, North-Y, Up-Z). Incorrect frame conversions lead to inverted or upside-down 3D meshes.")
    add_bullet_with_bold_prefix(doc, "Surface Normal Vector Computation", "Computing surface orientation using spatial gradients: Normal = normalized(cross_product(dP/du, dP/dv)). Normals dictate how light interacts with building roofs and terrain slopes.")
    add_bullet_with_bold_prefix(doc, "Heightfield Triangulation to GLTF/GLB", "Constructing a continuous polygonal surface by connecting adjacent 3D vertices into two triangles per quad cell. Exporting binary glTF (.glb) files containing bufferViews, accessors, vertex positions, UV coordinates, and Draco geometry compression.")
    
    add_styled_paragraph(doc, "The following Python snippet demonstrates unprojecting a 2D depth map into a 3D point cloud with vertex colors:")
    
    unproj_code = (
        "import numpy as np\n"
        "\n"
        "def depth_to_point_cloud(depth_map, rgb_img, fx, fy, cx, cy):\n"
        "    \"\"\"Unproject 2D depth and RGB into 3D point cloud (N, 3).\"\"\"\n"
        "    h, w = depth_map.shape\n"
        "    u, v = np.meshgrid(np.arange(w), np.arange(h))\n"
        "    \n"
        "    # Vectorized back-projection\n"
        "    z = depth_map.flatten()\n"
        "    x = (u.flatten() - cx) * z / fx\n"
        "    y = (v.flatten() - cy) * z / fy\n"
        "    \n"
        "    points_3d = np.stack([x, y, z], axis=-1)  # (H*W, 3)\n"
        "    colors_rgb = rgb_img.reshape(-1, 3) / 255.0\n"
        "    return points_3d, colors_rgb\n"
    )
    add_code_block(doc, unproj_code, "2D Depth Unprojection Blueprint")

    cv3d_rows = [
        ("First Principles of Computer Vision (Prof. Shree Nayar)", "Columbia University masterclass covering camera models, projective geometry, optics, and 3D vision.", "https://fpcv.cs.columbia.edu/"),
        ("Scratchapixel: 3D Math & Projective Geometry", "The premier interactive guide to camera projection matrices, ray tracing, and 3D coordinate spaces.", "https://www.scratchapixel.com/"),
        ("Open3D: A Modern Library for 3D Data Processing", "The industry standard Python/C++ library for point clouds, KD-trees, normal estimation, and surface meshing.", "http://www.open3d.org/"),
        ("Khronos glTF 2.0 Specification & Tutorials", "Complete documentation on binary GLB structure, buffers, Draco compression, and PBR materials.", "https://www.khronos.org/gltf/"),
        ("COLMAP Structure-from-Motion & Multi-View Stereo", "State-of-the-art open-source photogrammetry pipeline for multi-view 3D reconstruction from camera sets.", "https://colmap.github.io/")
    ]
    create_resource_table(doc, headers, cv3d_rows, table_widths, link_label="Open Resource")

    # -------------------------------------------------------------
    # 5. CUSTOM DECODER DESIGN & LOSS ENGINEERING
    # -------------------------------------------------------------
    h1 = doc.add_heading("5. Custom Decoder Design, Loss Functions & Training Engineering", level=1)
    style_heading(h1, font_size=16, color_rgb=(30, 58, 138), space_before=16, space_after=6)
    
    add_styled_paragraph(
        doc,
        "Achieving meter-accurate satellite elevation models requires replacing generic convolutional decoders with an Atrous Spatial Pyramid Pooling (ASPP) head and optimizing with a multi-objective loss function that balances relative scale, edge sharpness, and planar smoothness."
    )
    
    h2 = doc.add_heading("Key Theoretical Concepts", level=2)
    style_heading(h2, font_size=12, color_rgb=(13, 148, 136), space_before=6, space_after=4)
    
    add_bullet_with_bold_prefix(doc, "Atrous Spatial Pyramid Pooling (ASPP)", "Standard convolutions lose spatial resolution when expanding their receptive field. ASPP applies parallel atrous (dilated) convolutions at multiple sampling rates (e.g., 1, 6, 12, 18). This enables the decoder to capture both fine building parapets (small dilation) and broad terrain slopes (large dilation) without losing resolution.")
    add_bullet_with_bold_prefix(doc, "Scale-Invariant Logarithmic (SILog) Loss", "Standard L1/L2 losses produce blurry averages because a 2-meter error at a 10-meter rooftop is treated the same as a 2-meter error at a 500-meter mountain peak. SILog optimizes the logarithmic difference d_i = log(y_pred_i) - log(y_gt_i), penalizing relative percentage errors uniformly.")
    add_bullet_with_bold_prefix(doc, "Sobel Gradient & Edge Loss", "Computes spatial derivatives in X and Y directions using fixed Sobel kernels. Comparing the gradients of prediction and ground truth penalizes rounded building edges and forces the neural network to output sharp 90-degree vertical building walls.")
    add_bullet_with_bold_prefix(doc, "Surface Normal Consistency Loss", "Evaluates the dot product between predicted surface normal vectors and ground truth normal vectors: L_normal = 1 - dot(n_pred, n_gt). This forces building rooftops to be planar and eliminates high-frequency noise bumps on roads.")
    add_bullet_with_bold_prefix(doc, "Automatic Mixed Precision (AMP) & Checkpoint Management", "Using torch.cuda.amp.autocast and GradScaler to train in FP16/BF16 without underflow. Periodic training checkpoints retain AdamW momentum buffers (16.4 MB) to allow resuming, while production checkpoints strip optimizer buffers (5.49 MB) for rapid web serving.")
    
    add_styled_paragraph(doc, "Here is the PyTorch implementation of the ASPP multi-scale decoder head:")
    
    aspp_code = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "\n"
        "class ASPPHead(nn.Module):\n"
        "    \"\"\"Atrous Spatial Pyramid Pooling Decoder for Multi-Scale Elevation.\"\"\"\n"
        "    def __init__(self, in_channels=1024, out_channels=256, atrous_rates=(6, 12, 18)):\n"
        "        super().__init__()\n"
        "        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)\n"
        "        self.aspp_branches = nn.ModuleList([\n"
        "            nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False)\n"
        "            for r in atrous_rates\n"
        "        ])\n"
        "        self.global_pool = nn.Sequential(\n"
        "            nn.AdaptiveAvgPool2d(1),\n"
        "            nn.Conv2d(in_channels, out_channels, 1, bias=False)\n"
        "        )\n"
        "        self.project = nn.Sequential(\n"
        "            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, 1, bias=False),\n"
        "            nn.BatchNorm2d(out_channels),\n"
        "            nn.ReLU(inplace=True),\n"
        "            nn.Conv2d(out_channels, 1, kernel_size=3, padding=1)  # Final 1-channel DSM\n"
        "        )\n"
        "\n"
        "    def forward(self, x):\n"
        "        h, w = x.shape[2:]\n"
        "        res = [self.conv1x1(x)]\n"
        "        for branch in self.aspp_branches:\n"
        "            res.append(branch(x))\n"
        "        gp = F.interpolate(self.global_pool(x), size=(h, w), mode='bilinear', align_corners=False)\n"
        "        res.append(gp)\n"
        "        return self.project(torch.cat(res, dim=1))\n"
    )
    add_code_block(doc, aspp_code, "PyTorch ASPP Decoder Head Blueprint")

    train_rows = [
        ("Encoder-Decoder with Atrous Separable Convolutions (DeepLabV3+)", "The seminal paper establishing Atrous Spatial Pyramid Pooling for dense semantic feature extraction.", "https://arxiv.org/abs/1802.02611"),
        ("Depth Map Prediction using a Multi-Scale Deep Network (Eigen et al.)", "The foundational paper introducing the Scale-Invariant Logarithmic (SILog) loss for depth.", "https://arxiv.org/abs/1406.2283"),
        ("PyTorch Automatic Mixed Precision (AMP) Tutorial", "Official guide on using torch.cuda.amp for 2x faster training and 50% reduced GPU memory consumption.", "https://pytorch.org/docs/stable/amp.html"),
        ("AdamW & Cosine Annealing Learning Rate Schedules", "Understanding decoupled weight decay and smooth cosine learning rate decay with linear warmup.", "https://pytorch.org/docs/stable/optim.html")
    ]
    create_resource_table(doc, headers, train_rows, table_widths, link_label="Open Resource")

    # -------------------------------------------------------------
    # 6. SLIDING WINDOW INFERENCE & TILING
    # -------------------------------------------------------------
    h1 = doc.add_heading("6. Geospatial Tiling & Sliding Window Inference Engine", level=1)
    style_heading(h1, font_size=16, color_rgb=(30, 58, 138), space_before=16, space_after=6)
    
    add_styled_paragraph(
        doc,
        "Commercial satellite images arrive in massive dimensions (e.g. 8192 x 8192 pixels or larger). Passing an 8K image directly into a Vision Transformer requires hundreds of gigabytes of GPU VRAM, causing an immediate Out-Of-Memory (OOM) crash. Running inference requires a robust sliding-window tiling engine with seamless border blending."
    )
    
    h2 = doc.add_heading("Key Theoretical Concepts", level=2)
    style_heading(h2, font_size=12, color_rgb=(13, 148, 136), space_before=6, space_after=4)
    
    add_bullet_with_bold_prefix(doc, "Sliding Window Patching", "Decomposing high-resolution satellite orthomosaics into overlapping patches (e.g., 512x512 with a 64-pixel overlap margin).")
    add_bullet_with_bold_prefix(doc, "2D Hann / Cosine Taper Window Blending", "Why naive tile stitching causes visible grid lines: neural networks produce edge degradation at the boundaries of their receptive fields. Multiplying each tile prediction by a 2D Hann window ensures the center of the tile contributes 100% weight while edges taper smoothly to 0, completely eliminating visible seam lines.")
    add_bullet_with_bold_prefix(doc, "Accumulator Matrix & Normalization", "Maintaining two 2D arrays across the full scene: an Elevation Accumulator and a Weight Accumulator. Each patch prediction is weighted and added to the accumulators: Final_DSM = Elevation_Accumulator / Weight_Accumulator.")
    
    add_styled_paragraph(doc, "Here is the 2D Hann window weighting function used to guarantee seamless tile borders:")
    
    hann_code = (
        "import numpy as np\n"
        "\n"
        "def create_2d_hann_window(patch_h, patch_w):\n"
        "    \"\"\"Generate a 2D Hann (cosine-tapered) blending window.\"\"\"\n"
        "    hann_y = np.hanning(patch_h)\n"
        "    hann_x = np.hanning(patch_w)\n"
        "    window_2d = np.outer(hann_y, hann_x)\n"
        "    # Clamp minimum weight to avoid division by zero\n"
        "    return np.clip(window_2d, 1e-4, 1.0)\n"
    )
    add_code_block(doc, hann_code, "2D Hann Tiling Blending Blueprint")

    # -------------------------------------------------------------
    # 7. FULL-STACK WEBGIS & 3D VISUALIZATION
    # -------------------------------------------------------------
    h1 = doc.add_heading("7. Interactive 3D WebGIS Visualization (Next.js & Three.js)", level=1)
    style_heading(h1, font_size=16, color_rgb=(30, 58, 138), space_before=16, space_after=6)
    
    add_styled_paragraph(
        doc,
        "Delivering your 3D models to end users requires a high-performance web application. In Heimdall, this is built using Next.js 14, React Three Fiber (R3F), and Three.js to render interactive 3D terrain flythroughs at 60 FPS in any modern web browser."
    )
    
    h2 = doc.add_heading("Key Theoretical Concepts", level=2)
    style_heading(h2, font_size=12, color_rgb=(13, 148, 136), space_before=6, space_after=4)
    
    add_bullet_with_bold_prefix(doc, "Next.js App Router Architecture", "Server Components for static metadata and API endpoints vs. Client Components ('use client') for WebGL canvas mounting. Disabling SSR for 3D viewers using dynamic imports with { ssr: false } to prevent canvas hydration crashes.")
    add_bullet_with_bold_prefix(doc, "Three.js Scene Graph & Camera Controls", "Configuring PerspectiveCamera with proper near (0.1) and far (25,000) clipping planes. Configuring OrbitControls with damping, maxDistance limits, and smooth angle interpolation.")
    add_bullet_with_bold_prefix(doc, "Atmospheric Fog & Dynamic Falloff", "Why naive fog causes meshes to turn pitch black when zooming out: Standard linear fog cuts off at near distances. Implementing gentle wide-range falloffs (e.g. [1600, 14000]) allows full panoramic visibility of large satellite terrains without clipping.")
    add_bullet_with_bold_prefix(doc, "GLSL Shaders & Elevation Colormapping", "Writing custom vertex and fragment shaders in GLSL or mapping scientific color palettes (Turbo, Viridis, Terrain) to height values. Adding an interactive elevation exaggeration multiplier slider.")
    add_bullet_with_bold_prefix(doc, "Handling React Hydration Mismatches", "Solving real-world client-side DOM mutations caused by browser extensions (e.g., Dark Reader injecting inline SVG stroke styles). Using mounted state flags to ensure clean hydration.")
    
    web_rows = [
        ("Three.js Journey (by Bruno Simon)", "The absolute gold standard course for mastering WebGL, Three.js, custom shaders, lighting, performance, and 3D modeling.", "https://threejs-journey.com/"),
        ("React Three Fiber (R3F) Documentation", "Official documentation for declarative, reusable 3D graphics in React with ecosystem tools like @react-three/drei.", "https://r3f.docs.pmnd.rs/getting-started/introduction"),
        ("WebGL Fundamentals (From First Principles)", "In-depth interactive tutorial on how GPUs render triangles, vertex buffers, fragment shaders, and matrices.", "https://webglfundamentals.org/"),
        ("Next.js Official Documentation (App Router)", "Complete guide to Next.js 14, server actions, API routing, dynamic client components, and streaming.", "https://nextjs.org/docs")
    ]
    create_resource_table(doc, headers, web_rows, table_widths, link_label="Open Resource")

    # -------------------------------------------------------------
    # 8. STEP-BY-STEP 6-MONTH MASTERY ROADMAP
    # -------------------------------------------------------------
    h1 = doc.add_heading("8. Step-by-Step 6-Month Mastery Roadmap", level=1)
    style_heading(h1, font_size=16, color_rgb=(30, 58, 138), space_before=16, space_after=6)
    
    add_styled_paragraph(
        doc,
        "Here is a structured, chronological curriculum designed to guide you from foundational knowledge to coding the entire Heimdall pipeline independently."
    )
    
    roadmap_items = [
        ("Month 1: Deep Learning Fundamentals & PyTorch Core", 
         "Focus: Master PyTorch tensors, autograd, custom Dataset/DataLoader, convolution math, and backpropagation. Build a simple UNet on synthetic depth data. Complete 3Blue1Brown Neural Networks and start Stanford CS231n."),
        ("Month 2: Vision Transformers & Self-Supervised Geometry", 
         "Focus: Understand Patchify embeddings, multi-head self-attention, and DINOv2 visual representations. Learn how ViT outputs feature tokens at intermediate layers and how to reassemble them into 2D spatial maps."),
        ("Month 3: Geospatial Engineering & Satellite Datasets", 
         "Focus: Learn Rasterio, GDAL, and EPSG coordinate systems. Understand UTM projections, GSD, affine transforms, and DEM/DSM elevation data. Write scripts to inspect, crop, reproject, and tile GeoTIFF satellite scenes."),
        ("Month 4: 3D Math, Pinhole Cameras & Point Cloud Meshing", 
         "Focus: Study camera intrinsic matrices (K), pinhole projection, and back-projection equations. Unproject 2D depth arrays into 3D XYZ point clouds in Open3D. Write a Python script to triangulate heightfield grids and export binary GLB meshes."),
        ("Month 5: Custom ASPP Decoder & Multi-Objective Loss Training", 
         "Focus: Design the Atrous Spatial Pyramid Pooling (ASPP) decoder. Implement Scale-Invariant Logarithmic (SILog) loss, Sobel edge loss, and normal consistency loss. Train on Kaggle/Colab with AMP and save production checkpoints."),
        ("Month 6: Sliding-Window Tiling & Full-Stack 3D WebGIS App", 
         "Focus: Implement the sliding window inference engine with 2D Hann window blending for 8K satellite tiles. Build the Next.js frontend with React Three Fiber, custom OrbitControls, terrain flythrough, and live DSM downloading.")
    ]
    
    for title, desc in roadmap_items:
        add_bullet_with_bold_prefix(doc, title, desc)
        
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------
    # 9. MASTER RESOURCE CATALOG
    # -------------------------------------------------------------
    h1 = doc.add_heading("9. Master Resource Matrix: Curated Learning Catalog", level=1)
    style_heading(h1, font_size=16, color_rgb=(30, 58, 138), space_before=16, space_after=6)
    
    add_styled_paragraph(
        doc,
        "A master directory of the highest-rated free and open resources available across the web for each skill area."
    )
    
    master_table = doc.add_table(rows=12, cols=4)
    master_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    master_table.autofit = False
    m_widths = [Inches(1.2), Inches(2.2), Inches(1.8), Inches(1.3)]
    
    m_headers = ["Category", "Resource Name", "Creator / Publisher", "Format / Link"]
    for idx, text in enumerate(m_headers):
        cell = master_table.cell(0, idx)
        cell.width = m_widths[idx]
        set_cell_shading(cell, "1E3A8A")
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        p = cell.paragraphs[0]
        r = p.add_run(text)
        r.bold = True
        r.font.name = "Calibri"
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(255, 255, 255)
        
    master_catalog = [
        ("Deep Learning", "CS231n: Deep Learning for CV", "Stanford University", "https://youtube.com/playlist?list=PL3FW7PR474aMV2v1v70d9q9C61A8SjT8o"),
        ("Deep Learning", "Yannic Kilcher Paper Breakdowns", "Yannic Kilcher", "https://www.youtube.com/@YannicKilcher"),
        ("Deep Learning", "Attention Is All You Need (Transformer)", "Vaswani et al. (Google Brain)", "https://arxiv.org/abs/1706.03762"),
        ("3D Geometry", "First Principles of Computer Vision", "Prof. Shree Nayar (Columbia)", "https://www.youtube.com/c/FirstPrinciplesofComputerVision"),
        ("3D Graphics", "Three.js Full Tutorial", "freeCodeCamp / Bruno Simon", "https://www.youtube.com/watch?v=KM3K56Q0-dc"),
        ("Geospatial", "Remote Sensing & GIS Tutorials", "Earth Lab (CU Boulder)", "https://www.earthdatascience.org/courses/use-data-open-source-python/"),
        ("Book", "Multiple View Geometry in Computer Vision", "Hartley & Zisserman", "https://www.cambridge.org/core/books/multiple-view-geometry-in-computer-vision/1D787B2E2F80DAF4A950DBBE4F4DFB8C"),
        ("Repository", "Depth Anything V2 / V3 Official Repo", "TikTok / HKUST", "https://github.com/DepthAnything/Depth-Anything-V2"),
        ("Repository", "Metric3D: Monocular Metric Depth", "ByteDance Research", "https://github.com/YvanYin/Metric3D"),
        ("Repository", "PyTorch 3D Core Library", "Meta AI Research (FAIR)", "https://github.com/facebookresearch/pytorch3d"),
        ("Repository", "React Three Fiber & Drei", "Poimandres Collective", "https://github.com/pmndrs/react-three-fiber")
    ]
    
    for row_idx, (cat, name, creator, url) in enumerate(master_catalog, start=1):
        shading = "F8FAFC" if row_idx % 2 == 1 else "FFFFFF"
        for c_idx, width in enumerate(m_widths):
            cell = master_table.cell(row_idx, c_idx)
            cell.width = width
            set_cell_shading(cell, shading)
            set_cell_margins(cell, top=70, bottom=70, left=90, right=90)
            
        p0 = master_table.cell(row_idx, 0).paragraphs[0]
        r0 = p0.add_run(cat)
        r0.bold = True
        r0.font.name = "Calibri"; r0.font.size = Pt(9.0); r0.font.color.rgb = RGBColor(13, 148, 136)
        
        p1 = master_table.cell(row_idx, 1).paragraphs[0]
        r1 = p1.add_run(name)
        r1.font.name = "Calibri"; r1.font.size = Pt(9.0); r1.font.color.rgb = RGBColor(30, 58, 138)
        
        p2 = master_table.cell(row_idx, 2).paragraphs[0]
        r2 = p2.add_run(creator)
        r2.font.name = "Calibri"; r2.font.size = Pt(8.5); r2.font.color.rgb = RGBColor(55, 65, 81)
        
        p3 = master_table.cell(row_idx, 3).paragraphs[0]
        add_hyperlink(p3, url, "Open Resource", color_hex="2563EB", underline=True, bold=False, font_size_pt=8.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(16)
    
    # Final Sign-off / Advice
    callout_p = doc.add_paragraph()
    callout_p.paragraph_format.space_before = Pt(8)
    callout_p.paragraph_format.space_after = Pt(8)
    r_end = callout_p.add_run("Strategic Advice: ")
    r_end.bold = True
    r_end.font.color.rgb = RGBColor(30, 58, 138)
    r_end_body = callout_p.add_run(
        "Do not try to build every component on Day 1. Follow the 6-month roadmap step by step. Start by running inference on existing pretrained foundation models, understand how depth maps unproject to 3D point clouds in Open3D, practice writing GeoTIFFs in Rasterio, and then gradually build your custom ASPP decoder and web visualizer."
    )
    r_end_body.font.color.rgb = RGBColor(55, 65, 81)
    
    output_path = "/Users/aryanachary/.gemini/antigravity-ide/scratch/Heimdall/learning.docx"
    doc.save(output_path)
    print(f"Successfully generated master curriculum: {output_path}")

if __name__ == "__main__":
    main()
