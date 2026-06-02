# =============================================================
# pymol_tp53_hq.pml  v3  --  open-source PyMOL, clean geometry
# =============================================================

# Load only first biological unit (chain A protein + chain C DNA)
load 3kz8.pdb1, raw
remove solvent
remove (resname HOH or resname SO4 or resname GOL)

# Work with state 1 only (MODEL 1 of the NMR assembly)
create mono, raw, 1, 1
delete raw
remove (mono and not chain A+C)

select prot, mono and chain A
select dna,  mono and chain C
select zn,   mono and resname ZN

hide everything
bg_color white

# Assign secondary structure
dss prot

# ---- Quality settings ----
set ray_shadows,           1
set ray_trace_mode,        1
set ambient,               0.45
set direct,                0.65
set reflect,               0.10
set specular,              0.20
set cartoon_fancy_helices, 1
set cartoon_fancy_sheets,  1
set cartoon_smooth_loops,  1
set cartoon_tube_radius,   0.22
set stick_radius,          0.22
set sphere_scale,          0.50
set ray_opaque_background, 1
set antialias,             2

# ---- Base look ----
show cartoon, prot
show cartoon, dna
show spheres, zn

# Colour by secondary structure
color 0xF0F0F0, prot
color 0xE06060, (prot and ss h)
color 0xD8A840, (prot and ss s)
color 0x1188CC, dna
color 0xFFDD00, zn
set sphere_scale, 0.7, zn

# ---- Key residue selections ----
select structural_hs, prot and resi 135+175+176+238+242+245+282
select contact_mut,   prot and resi 248+273
select zinc_core,     prot and resi 135+176+238+242

# Use only DNA that is near the protein
select dna_near, (dna within 18 of prot)

# =============================================================
# VIEW 1: Overview -- protein+nearby DNA, well-framed
# =============================================================
show cartoon, prot
show sticks, structural_hs
show sticks, contact_mut
color 0xCC1199, structural_hs
color 0xFF8800, contact_mut
hide cartoon, (dna and not dna_near)
show cartoon, dna_near
color 0x1188CC, dna_near

orient (prot or dna_near)
zoom  (prot or dna_near), 6

ray 1400, 1050
png derived/figures/hq_1_overview_0.png, dpi=150

rotate y, 90
ray 1400, 1050
png derived/figures/hq_1_overview_90.png, dpi=150

rotate y, 90
ray 1400, 1050
png derived/figures/hq_1_overview_180.png, dpi=150

rotate y, 90
ray 1400, 1050
png derived/figures/hq_1_overview_270.png, dpi=150

# =============================================================
# VIEW 2: Hotspot surface -- grey protein, red hotspot patches
# =============================================================
hide sticks, structural_hs
hide sticks, contact_mut
hide cartoon, prot
show surface, prot
set transparency, 0.0
color 0xCCCCCC, prot
color 0xCC2222, (prot and resi 98+105+108+109+111+113+117+122+125+126+127+128+130+131+132+133+134+135+136+139+141+142+143+145+147+151+152+154+155+157+158+159+160+161+162+163+164+168+169+173+175+176+177+179+184+189+190+191+194+195+196+197+198+199+203+205+215+216+217+218+219+220+222+223+226+232+234+236+237+238+242+244+245+246+249+250+251+252+253+254+255+257+258+259+262+265+266+267+268+270+272+274+278+279+282+283)
color 0xFF8800, contact_mut

orient (prot or dna_near)
zoom  (prot or dna_near), 6

ray 1400, 1050
png derived/figures/hq_2_surface_0.png, dpi=150

rotate y, 90
ray 1400, 1050
png derived/figures/hq_2_surface_90.png, dpi=150

rotate y, 90
ray 1400, 1050
png derived/figures/hq_2_surface_180.png, dpi=150

rotate y, 90
ray 1400, 1050
png derived/figures/hq_2_surface_270.png, dpi=150

# =============================================================
# VIEW 3: Zinc-binding core (no DNA)
# =============================================================
hide surface
set transparency, 0
show cartoon, prot
hide everything, dna
hide everything, dna_near
color 0xF0F0F0, prot
color 0xE06060, (prot and ss h)
color 0xD8A840, (prot and ss s)

show sticks, zinc_core
show sticks, (prot and resi 175+245+249)
color 0xCC1199, zinc_core
color 0xCC1199, (prot and resi 175+245+249)
show spheres, zn
set sphere_scale, 0.9, zn
color 0xFFDD00, zn

label (zinc_core and name CA), "%s%s" % (resn, resi)
label (prot and resi 175 and name CA), "ARG175"
label (prot and resi 245 and name CA), "GLY245"
set label_size,   18
set label_color,  black

orient zinc_core
zoom zinc_core, 12

ray 1400, 1050
png derived/figures/hq_3_zinc_core.png, dpi=150

# =============================================================
# VIEW 4: DNA-contact interface (R248+R273 + DNA)
# =============================================================
label zinc_core, ""
label (prot and resi 175 and name CA), ""
label (prot and resi 245 and name CA), ""

show cartoon, dna_near
color 0x1188CC, dna_near

select contact_zone, (dna_near within 8 of contact_mut)
show sticks, contact_mut
show sticks, contact_zone
color 0xFF8800, contact_mut
color 0x0055AA, contact_zone

hide sticks, zinc_core
hide sticks, (prot and resi 175+245+249)

label (contact_mut and name CA), "%s%s" % (resn, resi)

orient (contact_mut or contact_zone)
zoom  (contact_mut or contact_zone), 9

ray 1400, 1050
png derived/figures/hq_4_contact.png, dpi=150

print "Done. 12 views saved to derived/figures/hq_*.png"
