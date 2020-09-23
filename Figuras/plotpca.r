  library(ggplot2)
  library(ggfortify)
  library(dplyr)
  library(ggbiplot)
  library(gridExtra)
  library(factoextra)
  setwd("/home/andbrav/Escritorio/TFG-INT/plots")
  PCA <- read.table("PCA.csv", header = TRUE, sep= ",")
  ISO <- read.table("ISOMAP.csv", header = TRUE, sep= ",")
  LE <- read.table("LAPLACIAN-EIGENMAPS.csv", header = TRUE, sep= ",")
  LLE <- read.table("LLE.csv", header = TRUE, sep= ",")
  tSNE <- read.table("TSNE.csv", header = TRUE, sep= ",")
  UMAP <- read.table("UMAP.csv", header = TRUE, sep= ",")
  
  
  bestPCA <- PCA %>%
    group_by(n_components) %>%
    filter(aggregate == max(aggregate)) %>%
    select(c(n_components,trustworthiness,continuity,normalized_stress,neighborhood_hit,shepard_diagram_correlation,aggregate))
  
  
  
  bestISO <- ISO %>%
    group_by(n_components) %>%
    filter(aggregate == max(aggregate)) %>%
    select(c(n_components,trustworthiness,continuity,normalized_stress,neighborhood_hit,shepard_diagram_correlation,aggregate))
  
  bestISO$normalized_stress[bestISO$normalized_stress > 1] =1
  
  bestLE <- LE %>%
    group_by(n_components) %>%
    filter(aggregate == max(aggregate)) %>%
    select(c(n_components,trustworthiness,continuity,normalized_stress,neighborhood_hit,shepard_diagram_correlation,aggregate))
  
  
  bestLLE <- LLE %>%
    group_by(n_components) %>%
    filter(aggregate == max(aggregate)) %>%
    select(c(n_components,trustworthiness,continuity,normalized_stress,neighborhood_hit,shepard_diagram_correlation,aggregate))
  
  
  besttSNE <- (tSNE %>%
    group_by(n_components) %>%
    filter(aggregate == max(aggregate))) %>%
    select(c(n_components,trustworthiness,continuity,normalized_stress,neighborhood_hit,shepard_diagram_correlation,aggregate))
  
  
  bestUMAP <- UMAP %>%
    group_by(n_components) %>%
    filter(aggregate == max(aggregate)) %>%
    select(c(n_components,trustworthiness,continuity,normalized_stress,neighborhood_hit,shepard_diagram_correlation,aggregate))
  
  
  
  
  
  
  
  dimension = 10
  ONED <- rbind(bestPCA[bestPCA$n_components==dimension,],
                bestISO[bestISO$n_components==dimension,],
                bestLE[bestLE$n_components==dimension,],
                bestLLE[bestLLE$n_components==dimension,],
                besttSNE[besttSNE$n_components==dimension,],
                bestUMAP[bestUMAP$n_components==dimension,])
  ONED2 <- ONED[,c(-1,-7)]
  row_names <- c("PCA","ISO","LE","LLE","tSNE","UMAP")
  rownames(ONED2) <- c("PCA","ISO","LE","LLE","tSNE","UMAP")
  ONED_dist<-dist(ONED2, method = "euclidean", diag = TRUE, upper = TRUE, p = 2)
  fit <- cmdscale(ONED_dist, eig = TRUE, k = 2)
  x <- fit$points[, 1]
  y <- fit$points[, 2]
  
  
  
  
  
  color.gradient <- function(x, colors=c("blue","green"), colsteps=100) {
    return( colorRampPalette(colors) (colsteps) [ findInterval(x, seq(min(x),max(x), length.out=colsteps)) ] )
  }
  
  
  
  colnames(ONED2) <- c("Trustworthiness","Continuity","Stress","Neighborhood hit","Correlation")
  pca_res <- prcomp(ONED2, scale. = TRUE)
  
#fviz_pca_ind(pca_res)
#  fviz_pca_var(pca_res)
    p2 <-autoplot(pca_res, data = ONED2,label=TRUE,
             label.size=3,
             colour=color.gradient(ONED$aggregate),
             label.colour="black",
             label.hjust = 1.5,
             label.vjust = 0,
             loadings = TRUE,
             loadings.label = TRUE,
             loadings.label.size = 3,
             loadings.label.vjust = -.75,
             loadings.label.hjust = 1
             ) + xlim(-1,1) + ylim(-.8,.8) +
    geom_point(aes(color=ONED$aggregate))+
    scale_color_gradient(low="blue", high="green",name = "Index")+
    ggtitle(paste("PCA para ",dimension,"dimensiones")) +
    geom_hline(yintercept=0, linetype="dashed", color = "grey")+
    geom_vline(xintercept=0, linetype="dashed", color = "grey")+
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
    grid.arrange(p1,p2, nrow = 1)




  





















