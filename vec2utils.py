
def vec2_dist(a,b): return (a-b).length
                    
def assign_nearest_center(points, centers):
    assignment = range(len(points))

    for i in range(len(points)):
        p = points[i]
        bestj = -1
        bestdist = 0.0
        for j in range(len(centers)):
            dist = vec2_dist(p,centers[j])
            if bestj == -1 or dist < bestdist:
                bestj = j
                bestdist = dist
        assignment[i] = bestj
    return assignment

