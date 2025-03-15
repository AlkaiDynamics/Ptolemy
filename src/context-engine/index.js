const fs = require('fs').promises;
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const { logger } = require('../../config/config');

class ContextEngine {
  constructor(temporalCore, storagePath) {
    this.temporalCore = temporalCore;
    this.storagePath = storagePath;
    this.relationshipsPath = path.join(storagePath, 'relationships');
    this.patternsPath = path.join(storagePath, 'patterns');
    this.insightsPath = path.join(storagePath, 'insights');
  }

  async initialize() {
    try {
      await fs.mkdir(this.relationshipsPath, { recursive: true });
      await fs.mkdir(this.patternsPath, { recursive: true });
      await fs.mkdir(this.insightsPath, { recursive: true });
      logger.info(`Context Engine initialized with storage path: ${this.storagePath}`);
    } catch (error) {
      logger.error(`Failed to initialize Context Engine: ${error.message}`);
      throw error;
    }
  }

  async storeRelationship(sourceEntity, targetEntity, relationshipType, metadata = {}) {
    const relationship = {
      id: uuidv4(),
      sourceEntity,
      targetEntity,
      relationshipType,
      metadata,
      timestamp: new Date().toISOString()
    };

    try {
      const relationshipFilePath = path.join(this.relationshipsPath, `${relationship.id}.json`);
      await fs.writeFile(relationshipFilePath, JSON.stringify(relationship, null, 2));
      
      // Record this as an event in the temporal core
      await this.temporalCore.recordEvent('relationship_created', {
        relationshipId: relationship.id,
        sourceEntity,
        targetEntity,
        relationshipType
      });
      
      logger.info(`Relationship stored: ${sourceEntity} -> ${targetEntity} (${relationshipType})`);
      return relationship;
    } catch (error) {
      logger.error(`Failed to store relationship: ${error.message}`);
      throw error;
    }
  }

  async storePattern(patternName, patternType, implementation, metadata = {}) {
    const pattern = {
      id: uuidv4(),
      name: patternName,
      type: patternType,
      implementation,
      metadata,
      timestamp: new Date().toISOString()
    };

    try {
      const patternFilePath = path.join(this.patternsPath, `${pattern.id}.json`);
      await fs.writeFile(patternFilePath, JSON.stringify(pattern, null, 2));
      
      // Record this as an event in the temporal core
      await this.temporalCore.recordEvent('pattern_stored', {
        patternId: pattern.id,
        patternName,
        patternType
      });
      
      logger.info(`Pattern stored: ${patternName} (${patternType})`);
      return pattern;
    } catch (error) {
      logger.error(`Failed to store pattern: ${error.message}`);
      throw error;
    }
  }

  async storeInsight(insightType, content, relevance = 1.0, metadata = {}) {
    const insight = {
      id: uuidv4(),
      type: insightType,
      content,
      relevance,
      metadata,
      timestamp: new Date().toISOString()
    };

    try {
      const insightFilePath = path.join(this.insightsPath, `${insight.id}.json`);
      await fs.writeFile(insightFilePath, JSON.stringify(insight, null, 2));
      
      // Record this as an event in the temporal core
      await this.temporalCore.recordEvent('insight_stored', {
        insightId: insight.id,
        insightType,
        relevance
      });
      
      logger.info(`Insight stored: ${insightType} (relevance: ${relevance})`);
      return insight;
    } catch (error) {
      logger.error(`Failed to store insight: ${error.message}`);
      throw error;
    }
  }

  async getRelationships(filters = {}) {
    try {
      const files = await fs.readdir(this.relationshipsPath);
      const relationships = [];

      for (const file of files) {
        if (path.extname(file) === '.json') {
          const filePath = path.join(this.relationshipsPath, file);
          const relationshipData = JSON.parse(await fs.readFile(filePath, 'utf8'));
          
          // Apply filters if any
          let includeRelationship = true;
          for (const [key, value] of Object.entries(filters)) {
            if (relationshipData[key] !== value) {
              includeRelationship = false;
              break;
            }
          }

          if (includeRelationship) {
            relationships.push(relationshipData);
          }
        }
      }

      return relationships;
    } catch (error) {
      logger.error(`Failed to get relationships: ${error.message}`);
      throw error;
    }
  }

  async getPatterns(filters = {}) {
    try {
      const files = await fs.readdir(this.patternsPath);
      const patterns = [];

      for (const file of files) {
        if (path.extname(file) === '.json') {
          const filePath = path.join(this.patternsPath, file);
          const patternData = JSON.parse(await fs.readFile(filePath, 'utf8'));
          
          // Apply filters if any
          let includePattern = true;
          for (const [key, value] of Object.entries(filters)) {
            if (patternData[key] !== value) {
              includePattern = false;
              break;
            }
          }

          if (includePattern) {
            patterns.push(patternData);
          }
        }
      }

      return patterns;
    } catch (error) {
      logger.error(`Failed to get patterns: ${error.message}`);
      throw error;
    }
  }

  async getInsights(filters = {}) {
    try {
      const files = await fs.readdir(this.insightsPath);
      const insights = [];

      for (const file of files) {
        if (path.extname(file) === '.json') {
          const filePath = path.join(this.insightsPath, file);
          const insightData = JSON.parse(await fs.readFile(filePath, 'utf8'));
          
          // Apply filters if any
          let includeInsight = true;
          for (const [key, value] of Object.entries(filters)) {
            if (insightData[key] !== value) {
              includeInsight = false;
              break;
            }
          }

          if (includeInsight) {
            insights.push(insightData);
          }
        }
      }

      return insights.sort((a, b) => b.relevance - a.relevance);
    } catch (error) {
      logger.error(`Failed to get insights: ${error.message}`);
      throw error;
    }
  }

  async getModelContext(task) {
    // Get relevant insights, patterns, and relationships for the task
    const insights = await this.getInsights();
    const patterns = await this.getPatterns();
    
    // Build context string
    let contextString = "# Project Context\n\n";
    
    // Add insights
    if (insights.length > 0) {
      contextString += "## Insights\n\n";
      for (const insight of insights.slice(0, 5)) { // Top 5 insights by relevance
        contextString += `- ${insight.content} (${insight.type}, relevance: ${insight.relevance})\n`;
      }
      contextString += "\n";
    }
    
    // Add patterns
    if (patterns.length > 0) {
      contextString += "## Implementation Patterns\n\n";
      for (const pattern of patterns.slice(0, 3)) { // Top 3 patterns
        contextString += `### ${pattern.name} (${pattern.type})\n\n\`\`\`\n${pattern.implementation}\n\`\`\`\n\n`;
      }
    }
    
    return contextString;
  }
}

module.exports = ContextEngine;
