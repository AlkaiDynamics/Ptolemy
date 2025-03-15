const axios = require('axios');
const { config, logger } = require('../../config/config');

class MultiModelRouter {
  constructor(contextEngine) {
    this.contextEngine = contextEngine;
    this.modelRegistry = {
      architect: { endpoint: 'openai', model: 'gpt-4', systemPrompt: 'Expert software architect.', temperature: 0.3 },
      implementer: { endpoint: 'openai', model: 'gpt-4', systemPrompt: 'Expert software developer.', temperature: 0.2 },
      reviewer: { endpoint: 'openai', model: 'gpt-4', systemPrompt: 'Expert code reviewer.', temperature: 0.3 },
      integrator: { endpoint: 'openai', model: 'gpt-4', systemPrompt: 'Integration specialist.', temperature: 0.4 }
    };
  }

  async routeTask(task, modelType, options = {}) {
    const modelConfig = this.modelRegistry[modelType];
    if (!modelConfig) throw new Error(`Unknown model type: ${modelType}`);

    const context = await this.contextEngine.getModelContext(task);
    const fullPrompt = `${context}\n\nTASK:\n${task}\n${options.additionalInstructions || ''}`;

    try {
      logger.info(`Routing task to ${modelType} model`);
      
      const response = await axios.post('https://api.openai.com/v1/chat/completions', {
        model: modelConfig.model,
        messages: [
          { role: "system", content: modelConfig.systemPrompt },
          { role: "user", content: fullPrompt }
        ],
        temperature: modelConfig.temperature
      }, { 
        headers: { 
          'Authorization': `Bearer ${config.openai.apiKey}`,
          'Content-Type': 'application/json'
        } 
      });

      logger.info(`Received response from ${modelType} model`);
      return response.data.choices[0].message.content;
    } catch (error) {
      logger.error(`Error routing task to ${modelType} model: ${error.message}`);
      throw error;
    }
  }

  registerModel(modelType, config) {
    this.modelRegistry[modelType] = config;
    logger.info(`Registered model: ${modelType}`);
  }

  async routeMultiStage(task, stages, options = {}) {
    let currentContext = task;
    let results = [];

    for (const stage of stages) {
      try {
        const result = await this.routeTask(
          currentContext, 
          stage.modelType, 
          { additionalInstructions: stage.instructions, ...options }
        );
        
        results.push({ stage: stage.name, output: result });
        currentContext += `\n\nPrevious output:\n${result}\n\nNext task:\n${stage.nextPrompt || ''}`;
      } catch (error) {
        logger.error(`Error in stage ${stage.name}: ${error.message}`);
        if (!options.continueOnError) throw error;
      }
    }

    return results;
  }
}

module.exports = MultiModelRouter;
