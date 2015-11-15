import shapegen

(G, shapes) = shapegen.create_clump_of_shapes(200, 200, 3, 3)
G.save_png("final.png")
