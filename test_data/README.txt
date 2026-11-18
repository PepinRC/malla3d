Test files (OBJ, millimeters, Z up, body length ~500 mm)

front_zone_input_lowpoly.obj   Front fender/hood area of the input low-poly (Tripo Smart Mesh), 205 faces.
full_body_input_lowpoly.obj    Whole body shell of the same input, 1,635 faces (mixed quads/triangles).
front_zone_our_cage.obj        Our best attempt for the front area: 914 quads (guided QuadRemesh + loops snapped to fold lines).
front_zone_fold_lines.json     Fold lines of the input zone (61 polylines, mm).
front_zone_our_cage_creases.json  Crease edges of that cage as vertex index pairs (0-based).

Goal: a lightweight, editable SubD cage (all quads, even sizes, no long thin faces) that keeps every
plane, chamfer (1-2 mm wide) and step of the input when subdivided with those edges creased.
