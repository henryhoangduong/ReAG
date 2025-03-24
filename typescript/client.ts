import { QueryResult } from "./client";
import {
  generateText,
  LanguageModel,
  GenerateObjectResult,
  generateObject,
} from "ai";
import { RESPONSE_SCHEMA } from "./schema";
import { z } from "zod";
export interface ClientOptions {
  model: LanguageModel;
  system: string;
  batchSize: number;
  schema?: z.ZodSchema;
}

export interface Document {
  name: string;
  content: string;
  metadata?: Record<string, string | number>;
}

export interface QueryResult<T> {
  results: {
    relevant: T[];
    irrelevant?: boolean;
  };
  doucment: Document;
}

export interface MetaDataFilter {
  key: string;
  value: string | number;
  operator?:
    | "equals"
    | "contains"
    | "startsWith"
    | "endsWith"
    | "notEquals"
    | "greaterThan"
    | "lessThan"
    | "greaterThanOrEqual"
    | "lessThanOrEqual"
    | "regex";
}

const DEFAULT_BATCH_SIZE = 20;

export class ReagClient {
  private readonly model: LanguageModel;
  private readonly system: string;
  private readonly batchSize: number;
  private readonly schema: z.ZodSchema;

  /**
   * Construct a new Client instance
   * @param options Configuration options for the Client.
   */
  constructor(options: ClientOptions) {
    this.model = options.model;
    this.system = options.system;
    this.batchSize = options.batchSize || DEFAULT_BATCH_SIZE;
    this.schema = options.schema || RESPONSE_SCHEMA;
  }

  /**
   * Filters documents based on metadata criteria
   */
  private filterDocumentByMetadata(
    documents: Document[],
    filter?: MetaDataFilter[],
  ): Document[] {
    if (!filter?.length) return documents;
    return documents.filter((doc) => {
      return filter.every((filter) => {
        const metaDataValue = doc.metadata?.[filter.key];
        if (!metaDataValue) return false;
        if (
          typeof metaDataValue == "string" &&
          typeof filter.value == "string"
        ) {
          switch (filter.operator) {
            case "contains":
              return metaDataValue.includes(filter.value);
            case "startsWith":
              return metaDataValue.startsWith(filter.value);
            case "endsWith":
              return metaDataValue.endsWith(filter.value);
            case "regex":
              return new RegExp(filter.value).test(metaDataValue);
          }
        }
        switch (filter.operator) {
          case "equals":
            return metaDataValue === filter.value;
          case "notEquals":
            return metaDataValue !== filter.value;
          case "greaterThan":
            return metaDataValue > filter.value;
          case "lessThan":
            return metaDataValue < filter.value;
          case "greaterThanOrEqual":
            return metaDataValue >= filter.value;
          case "lessThanOrEqual":
            return metaDataValue <= filter.value;
          default:
            return metaDataValue === filter.value;
        }
      });
    });
  }
  /**
   * Executes a query on the assigned language model with document batching
   */
  async query<T extends z.ZodType>(
    prompt: string,
    documents: Document[],
    options?: {
      filter?: MetaDataFilter[];
    },
  ): Promise<QueryResult<z.infer<T>>[]> {
    try {
      const filterDocuments = this.filterDocumentByMetadata(
        documents,
        options?.filter,
      );
      const formatDoc = (doc: Document) =>
        `Name: ${doc.name}\nMetadata: ${JSON.stringify(
          doc.metadata,
        )}\nContent: ${doc.content}`;
      const batches = Array.from(
        {
          length: Math.ceil(filterDocuments.length / this.batchSize),
        },
        (_, i) => {
          filterDocuments.slice(i * this.batchSize, (i + 1) * this.batchSize);
        },
      );

      const batchResults = await Promise.all(
        batches.map(async (batch) => {
          const batchResponses = await Promise.all(
            batch.map(async (document) => {
              const system = `${
                this.system
              }\n\n# Available source\n\n${formatDoc(document)}`;
              const response = await generateObject({
                model: this.model,
                system,
                prompt,
                schema: this.schema,
              });

              return {
                response,
                document,
              };
            }),
          );
          return batchResponses;
        }),
      );

      const results = batchResults.flat().map(({ response, document }) => ({
        ...response.object,
        document,
      }));

      return results;
    } catch (error) {
      throw new Error(`Query failed: ${error}`);
    }
  }
}
