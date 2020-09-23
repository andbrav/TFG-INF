library(ggplot2)
library(gridExtra)
setwd("/home/andbrav/Escritorio/TFG-INT/plots")
PCA <- read.table("PCA.csv", header = TRUE, sep= ",")
ISO <- read.table("ISOMAP.csv", header = TRUE, sep= ",")
LE <- read.table("LAPLACIAN-EIGENMAPS.csv", header = TRUE, sep= ",")
LLE <- read.table("LLE.csv", header = TRUE, sep= ",")
tSNE <- read.table("TSNE.csv", header = TRUE, sep= ",")
UMAP <- read.table("UMAP.csv", header = TRUE, sep= ",")


pcaplot <- ggplot(PCA, aes(x=aggregate)) + 
  xlim(0,1) + 
  geom_density(fill="#69b3a2", color="#e9ecef", alpha=0.8) + 
  ggtitle(paste0("PCA")) +
  xlab("Índice") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
pcaplot

isoplot <- ggplot(ISO, aes(x=aggregate)) + 
  xlim(0,1) + 
  geom_density(fill="#69b3a2", color="#e9ecef", alpha=0.8) + 
  ggtitle(paste0("ISOMAP")) +
  xlab("Índice") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
isoplot

leplot <- ggplot(LE, aes(x=aggregate)) + 
  xlim(0,1) + 
  geom_density(fill="#69b3a2", color="#e9ecef", alpha=0.8) + 
  ggtitle(paste0("Laplacian eigenmaps")) +
  xlab("Índice") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
leplot

lleplot <- ggplot(LLE, aes(x=aggregate)) + 
  xlim(0,1) + 
  geom_density(fill="#69b3a2", color="#e9ecef", alpha=0.8) + 
  ggtitle(paste0("Locally linear embedding")) +
  xlab("Índice") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
lleplot

tsneplot <- ggplot(tSNE, aes(x=aggregate)) + 
  xlim(0,1) + 
  geom_density(fill="#69b3a2", color="#e9ecef", alpha=0.8) + 
  ggtitle(paste0("tSNE")) +
  xlab("Índice") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
tsneplot

umapplot <- ggplot(UMAP, aes(x=aggregate)) + 
  xlim(0,1) + 
  geom_density(fill="#69b3a2", color="#e9ecef", alpha=0.8) + 
  ggtitle(paste0("UMAP")) +
  xlab("Índice") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
umapplot


grid.arrange(pcaplot, isoplot, leplot, lleplot, tsneplot, umapplot, nrow = 2)