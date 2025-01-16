import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np

class PRFQueryExpander(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Term importance scorer
        self.term_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Query expansion generator
        self.expansion_generator = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, query_terms, feedback_docs, n_terms=10):
        # Embed query terms
        query_embeds = self.embedding(query_terms)  # [Q, E]
        query_ctx = query_embeds.mean(dim=0)  # [E]
        
        # Process feedback documents
        doc_terms = []
        doc_weights = defaultdict(float)
        
        for doc in feedback_docs:
            doc_embeds = self.embedding(doc)  # [D, E]
            doc_ctx = doc_embeds.mean(dim=0)  # [E]
            
            # Score terms based on query and document context
            for term_idx, term_embed in enumerate(doc_embeds):
                term_score = self.term_scorer(
                    torch.cat([term_embed, query_ctx])
                ).item()
                doc_weights[doc[term_idx].item()] += term_score
        
        # Select top expansion terms
        expansion_terms = sorted(
            doc_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_terms]
        
        # Generate expansion embeddings
        expansion_embeds = self.embedding(
            torch.tensor([t[0] for t in expansion_terms])
        )
        
        # Combine with original query
        expanded_query = self.expansion_generator(
            torch.cat([
                query_ctx,
                expansion_embeds.mean(dim=0),
                query_ctx * expansion_embeds.mean(dim=0)
            ])
        )
        
        return expanded_query, expansion_terms

class RLDocumentRetriever(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128):
        super().__init__()
        self.query_expander = PRFQueryExpander(vocab_size, embedding_dim)
        
        # Document scorer
        self.doc_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Value network for RL
        self.value_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, query, candidates, feedback_docs=None, temperature=1.0):
        # Expand query if feedback documents are provided
        if feedback_docs is not None:
            query, expansion_terms = self.query_expander(query, feedback_docs)
        else:
            expansion_terms = None
            
        # Score candidate documents
        doc_scores = []
        for doc in candidates:
            doc_embed = self.embedding(doc).mean(dim=0)
            score = self.doc_scorer(
                torch.cat([query, doc_embed])
            )
            doc_scores.append(score)
        
        doc_scores = torch.stack(doc_scores)
        
        # Apply temperature scaling
        probs = F.softmax(doc_scores / temperature, dim=0)
        
        # Estimate state value for RL
        value = self.value_net(query)
        
        return probs, value, expansion_terms

class PRFRetrievalAgent:
    def __init__(self, vocab_size, embedding_dim=300, lr=1e-4):
        self.retriever = RLDocumentRetriever(vocab_size, embedding_dim)
        self.optimizer = torch.optim.Adam(self.retriever.parameters(), lr=lr)
        
    def select_documents(self, query, candidates, feedback_docs=None):
        with torch.no_grad():
            probs, _, expansion_terms = self.retriever(
                query, candidates, feedback_docs
            )
            return probs.numpy(), expansion_terms
    
    def update(self, experiences, gamma=0.99, entropy_coef=0.01):
        """Update policy using PPO with PRF"""
        total_loss = 0
        
        for batch in experiences:
            query, candidates, feedback_docs, action, reward, next_query = batch
            
            # Forward pass with PRF
            probs, value, _ = self.retriever(query, candidates, feedback_docs)
            
            # Compute advantages
            next_value = self.retriever.value_net(next_query)
            advantage = reward + gamma * next_value - value
            
            # PPO policy loss
            old_probs = probs.detach()
            ratio = probs / old_probs
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(value, reward + gamma * next_value)
            
            # Entropy bonus
            entropy = -(probs * torch.log(probs)).sum()
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
            total_loss += loss

            # Update networks
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return total_loss.item()

def train_prf_retriever(agent, env, n_episodes=1000, feedback_docs_per_query=3):
    """Train the PRF-enhanced retriever"""
    for episode in range(n_episodes):
        experiences = []
        episode_reward = 0
        query = env.reset()
        feedback_docs = None
        
        done = False
        while not done:
            # Get candidate documents
            candidates = env.get_candidate_documents()
            
            # Select documents using current policy and PRF
            probs, expansion_terms = agent.select_documents(
                query, candidates, feedback_docs
            )
            selected_idx = np.random.choice(len(candidates), p=probs)
            
            # Take action and get feedback
            next_query, reward, done, info = env.step(selected_idx)
            
            # Update feedback documents
            if feedback_docs is None:
                feedback_docs = []
            if len(feedback_docs) < feedback_docs_per_query:
                feedback_docs.append(candidates[selected_idx])
            
            # Store experience
            experiences.append((
                query, candidates, feedback_docs.copy(),
                selected_idx, reward, next_query
            ))
            
            episode_reward += reward
            query = next_query
        
        # Update policy
        loss = agent.update(experiences)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Loss: {loss:.4f}")
