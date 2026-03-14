import { NvlOptions } from '@neo4j-nvl/base';
import { GraphType, OptionType } from '../types';

export const document = `+ [docs]`;

export const chunks = `+ collect { MATCH p=(c)-[:NEXT_CHUNK]-() RETURN p } // chunk-chain
+ collect { MATCH p=(c)-[:SIMILAR]-() RETURN p } // similar-chunks`;

export const entities = `+ collect { OPTIONAL MATCH (c:Chunk)-[:HAS_ENTITY]->(e), p=(e)-[*0..1]-(:!Chunk) RETURN p}`;

export const docEntities = `+ [docs] 
+ collect { MATCH (c:Chunk)-[:HAS_ENTITY]->(e), p=(e)--(:!Chunk) RETURN p }`;

export const docChunks = `+[chunks]
+collect {MATCH p=(c)-[:FIRST_CHUNK]-() RETURN p} //first chunk
+ collect { MATCH p=(c)-[:NEXT_CHUNK]-() RETURN p } // chunk-chain
+ collect { MATCH p=(c)-[:SIMILAR]-() RETURN p } // similar-chunk`;

export const chunksEntities = `+ collect { MATCH p=(c)-[:NEXT_CHUNK]-() RETURN p } // chunk-chain

+ collect { MATCH p=(c)-[:SIMILAR]-() RETURN p } // similar-chunks
//chunks with entities
+ collect { OPTIONAL MATCH p=(c:Chunk)-[:HAS_ENTITY]->(e)-[*0..1]-(:!Chunk) RETURN p }`;

export const docChunkEntities = `+[chunks]
+collect {MATCH p=(c)-[:FIRST_CHUNK]-() RETURN p} //first chunk
+ collect { MATCH p=(c)-[:NEXT_CHUNK]-() RETURN p } // chunk-chain
+ collect { MATCH p=(c)-[:SIMILAR]-() RETURN p } // similar-chunks
//chunks with entities
+ collect { OPTIONAL MATCH p=(c:Chunk)-[:HAS_ENTITY]->(e)-[*0..1]-(:!Chunk) RETURN p }`;
export const APP_SOURCES =
  process.env.REACT_APP_SOURCES !== ''
    ? process.env.REACT_APP_SOURCES?.split(',')
    : ['gcs', 's3', 'local', 'wiki', 'youtube', 'web'];
export const llms =
  process.env?.LLM_MODELS?.trim() != ''
    ? process.env.LLM_MODELS?.split(',')
    : [
        'diffbot',
        'openai-gpt-3.5',
        'openai-gpt-4o',
        'gemini-1.0-pro',
        'gemini-1.5-pro',
        'azure_ai_gpt_35',
        'azure_ai_gpt_4o',
        'ollama_llama3',
        'groq_llama3_70b',
        'anthropic_claude_3_5_sonnet',
        'fireworks_llama_v3_70b',
        'bedrock_claude_3_5_sonnet',
      ];

export const defaultLLM = llms?.includes('openai-gpt-3.5')
  ? 'openai-gpt-3.5'
  : llms?.includes('gemini-1.0-pro')
  ? 'gemini-1.0-pro'
  : 'diffbot';
export const chatModes =
  process.env?.CHAT_MODES?.trim() != '' ? process.env.CHAT_MODES?.split(',') : ['vector', 'graph', 'graph+vector'];
export const chunkSize = process.env.CHUNK_SIZE ? parseInt(process.env.CHUNK_SIZE) : 1 * 1024 * 1024;
export const timeperpage = process.env.TIME_PER_PAGE ? parseInt(process.env.TIME_PER_PAGE) : 50;
export const timePerByte = 0.2;
export const largeFileSize = process.env.LARGE_FILE_SIZE ? parseInt(process.env.LARGE_FILE_SIZE) : 5 * 1024 * 1024;
export const NODES_OPTIONS = [
  {
    label: 'Person',
    value: 'Person',
  },
  {
    label: 'Organization',
    value: 'Organization',
  },
  {
    label: 'Event',
    value: 'Event',
  },
];

export const RELATION_OPTIONS = [
  {
    label: 'WORKS_AT',
    value: 'WORKS_AT',
  },
  {
    label: 'IS_CEO',
    value: 'IS_CEO',
  },
  {
    label: 'HOSTS_EVENT',
    value: 'HOSTS_EVENT',
  },
];

export const queryMap: {
  Document: string;
  Chunks: string;
  Entities: string;
  DocEntities: string;
  DocChunks: string;
  ChunksEntities: string;
  DocChunkEntities: string;
} = {
  Document: 'document',
  Chunks: 'chunks',
  Entities: 'entities',
  DocEntities: 'docEntities',
  DocChunks: 'docChunks',
  ChunksEntities: 'chunksEntities',
  DocChunkEntities: 'docChunkEntities',
};

export const tooltips = {
  generateGraph: '选择一个或多个（新）文件转化为图谱',
  deleteFile: '选择一个或多个要删除的文件',
  showGraph: '选择一个或多个文件预览生成的图谱',
  bloomGraph: '打开Neo4j Bloom进行高级图谱交互和探索',
  deleteSelectedFiles: '要删除的文件',
  documentation: '文档',
  github: 'GitHub问题',
  theme: '亮色/暗色模式',
  settings: '实体图谱提取设置',
  chat: '询问已处理文档的问题',
  sources: '上传不同格式的文件',
  deleteChat: '删除',
  maximise: '最大化',
  copy: '复制到剪贴板',
  copied: '已复制',
  stopSpeaking: '停止朗读',
  textTospeech: '文本转语音',
  createSchema: '通过文本创建自己的模式',
  useExistingSchema: '使用数据库中已有的模式',
  clearChat: '清除聊天记录',
  continue: '继续',
  clearGraphSettings: '允许用户移除设置',
};

export const buttonCaptions = {
  exploreGraphWithBloom: '使用Bloom探索图谱',
  showPreviewGraph: '预览图谱',
  deleteFiles: '删除文件',
  generateGraph: '生成图谱',
  dropzoneSpan: '拖放或浏览文档、图像、非结构化文本',
  youtube: '油管视频',
  gcs: 'GCS',
  amazon: '亚马逊S3',
  noLables: '数据库中未找到标签',
  dropYourCreds: '在此处放置您的Neo4j凭据文件',
  analyze: '分析文本以提取图谱模式',
  connect: '连接',
  disconnect: '断开连接',
  submit: '提交',
  connectToNeo4j: '连接到Neo4j',
  cancel: '取消',
  details: '详情',
  continueSettings: '继续',
  clearSettings: '清除设置',
  ask: '提问',
};

export const taskParam: string[] = ['update_similarity_graph', 'create_fulltext_index', 'create_entity_embedding'];

export const nvlOptions: NvlOptions = {
  allowDynamicMinZoom: true,
  disableWebGL: true,
  maxZoom: 3,
  minZoom: 0.05,
  relationshipThreshold: 0.55,
  useWebGL: false,
  instanceId: 'graph-preview',
  initialZoom: 1,
};

export const mouseEventCallbacks = {
  onPan: true,
  onZoom: true,
  onDrag: true,
};

export const graphQuery: string = queryMap.DocChunkEntities;
export const graphView: OptionType[] = [
  { label: '词法图谱', value: queryMap.DocChunks },
  { label: '实体图谱', value: queryMap.Entities },
  { label: '知识图谱', value: queryMap.DocChunkEntities },
];
export const intitalGraphType: GraphType[] = ['Document', 'Entities', 'Chunk'];
export const knowledgeGraph = '知识图谱';
export const lexicalGraph = '词法图谱';
export const entityGraph = '实体图谱';

export const appLabels = {
  ownSchema: '或定义您自己的模式',
  predefinedSchema: '选择预定义模式',
};
